#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <utility>

// close() drains queued work; cancel() discards it. Both wake blocked callers.
template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(size_t capacity) : capacity_(capacity) {
        if (!capacity) throw std::invalid_argument("Queue capacity must be positive");
    }
    bool push(T value) {
        std::unique_lock<std::mutex> lock(mutex_);
        changed_.wait(lock, [&] { return closed_ || items_.size() < capacity_; });
        if (closed_) return false;
        items_.push_back(std::move(value));
        changed_.notify_all();
        return true;
    }
    // Only used for live previews/jobs: replace queued, unstarted work.
    size_t pushLatest(T value) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) return 0;
        size_t removed = 0;
        while (items_.size() >= capacity_) { items_.pop_front(); ++removed; }
        items_.push_back(std::move(value));
        changed_.notify_all();
        return removed;
    }
    bool pop(T& value) {
        std::unique_lock<std::mutex> lock(mutex_);
        changed_.wait(lock, [&] { return closed_ || !items_.empty(); });
        return take(value);
    }
    bool popFor(T& value, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        changed_.wait_for(lock, timeout, [&] { return closed_ || !items_.empty(); });
        return take(value);
    }
    bool finished() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return closed_ && items_.empty();
    }
    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        changed_.notify_all();
    }
    void cancel() {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        items_.clear();
        changed_.notify_all();
    }
private:
    bool take(T& value) {
        if (items_.empty()) return false;
        value = std::move(items_.front());
        items_.pop_front();
        changed_.notify_all();
        return true;
    }
    const size_t capacity_;
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    std::deque<T> items_;
    bool closed_ = false;
};
