---
layout: post
title: Indexed Priority Queue Explained with the Python implementation!
---

Priority Queue is a an abstract queue-like data structure where each element has a priority. Consider a problem where we have many objects and at each step, we only need the element with the highest (or lowest based on the type) priority. In this situation, priority queue is the data structure to use. But what if at the middle of the program, priority of an object changes? Indexed Priority Queue gives us the ability to change the priority of an element without having to go through all the elements.  

Before, going throgh the details on how to implement a Indexed Priority Queue, let's see how we can implement a priority queue. Priority queues are usually implemented with heaps. A heap is a tree-based data structure where for a node C, the key of C is grater than the the key of C's parent (for MinPQ). To have an efficient access to a node's childs or parent, an array is usualy used to represent the tree where if the key of node i is pq[i], the childs are pq[2\*i] and pq[2\*i+1].


!["Heap Img"](https://raw.githubusercontent.com/mkhoshpa/mkhoshpa.github.io/master/images/Heap.png?style=centerme)

To ensure that the node with the highest priority is always on top of the tree, heaps use swim and sink methods. Swim method ensures that the node does not have higher priority than it's parent and sink makes sure that the children of the node has lower priority.


In the case of MaxPQ:
```python
class PQ:
    def __init__(self, N):
        self.arr = [None]
        assert type(N) is int
        self.N = N
        
    def __sink(self, idx):
        assert type(idx) is int
        assert idx > 0 and idx < N+1
        while 2 * idx < self.N:
            child = 2 * idx
            if child + 1 < N:
                if self.arr[child + 1 ] > self.arr[child]:
                    child = child + 1
            if arr[child] > arr[idx]:
                self.arr[child], self.arr[idx] = self.arr[idx], self.arr[child]
                idx = child
            else:
                break

    def __swim(idx):
        assert type(idx) is int
        assert idx > 0 and idx < N+1
        while idx // 2 > 0:
            p = idx // 2
            if self.arr[idx]> self.arr[p]:
                self.arr[idx], self.arr[p] = self.arr[p], self.arr[idx]
                idx = p
            else:
                break
                
     def isEmpty():
        return len(self.arr) == 1
        
     def insert(key):
        self.arr.append(key)
        self.__swim(len(arr) - 1)
      
     def remove():
        if not self.isEmpty():
            out = self.arr[1]
            self.arr[1] = self.arr.pop()
            self.__sink(1)
            return out
        raise Exception('PQ is empty')
        
        
```

While priority queues areuseful in many applications, they do not support changing the priority of an element in logarithmic time. However, with a minor chamge, we can add indexing to the priority queue. 


!["Heap Img"](https://raw.githubusercontent.com/mkhoshpa/mkhoshpa.github.io/master/images/indexedPQ.png?style=centerme) 


In an Indexed priority queue, we store the information in three arrays. The keys (key), the heap represention of the priority queue (pq), and in array containing the index of each key in the heap representation (qp). So we have qp[pq[i]] = i.

