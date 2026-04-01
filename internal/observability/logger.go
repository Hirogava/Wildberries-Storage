package observability

import (
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"time"
)

type Entry struct {
	Timestamp time.Time `json:"timestamp"`
	Level     string    `json:"level"`
	Component string    `json:"component"`
	Message   string    `json:"message"`
}

type Logger struct {
	mu          sync.RWMutex
	maxSize     int
	entries     []Entry
	subscribers map[int]chan Entry
	nextID      int
}

func NewLogger(maxSize int) *Logger {
	if maxSize <= 0 {
		maxSize = 300
	}

	return &Logger{
		maxSize:     maxSize,
		entries:     make([]Entry, 0, maxSize),
		subscribers: make(map[int]chan Entry),
	}
}

func (l *Logger) Info(component, format string, args ...any) {
	l.add("INFO", component, format, args...)
}

func (l *Logger) Warn(component, format string, args ...any) {
	l.add("WARN", component, format, args...)
}

func (l *Logger) Error(component, format string, args ...any) {
	l.add("ERROR", component, format, args...)
}

func (l *Logger) Entries(limit int) []Entry {
	l.mu.RLock()
	defer l.mu.RUnlock()

	if limit <= 0 || limit > len(l.entries) {
		limit = len(l.entries)
	}

	start := len(l.entries) - limit
	result := make([]Entry, limit)
	copy(result, l.entries[start:])
	return result
}

func (l *Logger) Subscribe() (<-chan Entry, func()) {
	l.mu.Lock()
	defer l.mu.Unlock()

	id := l.nextID
	l.nextID++

	ch := make(chan Entry, 32)
	l.subscribers[id] = ch

	cancel := func() {
		l.mu.Lock()
		defer l.mu.Unlock()

		subscriber, ok := l.subscribers[id]
		if !ok {
			return
		}

		delete(l.subscribers, id)
		close(subscriber)
	}

	return ch, cancel
}

func MarshalEntry(entry Entry) ([]byte, error) {
	return json.Marshal(entry)
}

func (l *Logger) add(level, component, format string, args ...any) {
	entry := Entry{
		Timestamp: time.Now().UTC(),
		Level:     level,
		Component: component,
		Message:   fmt.Sprintf(format, args...),
	}

	l.mu.Lock()
	if len(l.entries) == l.maxSize {
		copy(l.entries, l.entries[1:])
		l.entries[len(l.entries)-1] = entry
	} else {
		l.entries = append(l.entries, entry)
	}

	subscribers := make([]chan Entry, 0, len(l.subscribers))
	for _, subscriber := range l.subscribers {
		subscribers = append(subscribers, subscriber)
	}
	l.mu.Unlock()

	for _, subscriber := range subscribers {
		select {
		case subscriber <- entry:
		default:
		}
	}

	log.Printf("[%s] [%s] %s", entry.Level, entry.Component, entry.Message)
}
