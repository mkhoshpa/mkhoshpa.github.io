---
layout: post
title: Indexed Priority Queue Explained with the Python implementation!
---
While priority queues are useful in many applications, they do not support changing the priority of an element in logarithmic time. However, with a minor change, we can add indexing to the priority queue. 


!["Heap Img"](https://raw.githubusercontent.com/mkhoshpa/mkhoshpa.github.io/master/images/indexedPQ.png?style=centerme) 



Priority Queue is an abstract queue-like data structure where each element has a priority. Consider a problem where we have many objects, and at each step, we only need the element with the highest (or lowest based on the type) priority. In this situation, a priority queue is the data structure to use. But what if in the middle of the program, the priority of an object changes? Indexed Priority Queue gives us the ability to change the priority of an element without having to go through all the elements.  

Before going through the details on how to implement an Indexed Priority Queue, let's see how we can implement a priority queue. Priority queues are usually implemented with heaps. A heap is a tree-based data structure where for node C, C's key is greater than the key of C's parent (for MinPQ). To have efficient access to the node's children or the node's parent, an array is usually used to represent the tree where if the key of node i is pq[i], the children are pq[2\*i] and pq[2\*i+1].


!["Heap Img"](https://raw.githubusercontent.com/mkhoshpa/mkhoshpa.github.io/master/images/Heap.png?style=centerme)

Heaps use swim and sink methods to ensure that the node with the highest priority is always on top of the tree. The swim method ensures that the node does not have a higher priority than its parent, and the sink method makes sure that the node's children have lower priority.


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

While priority queues are useful in many applications, they do not support changing the priority of an element in logarithmic time. However, with a minor change, we can add indexing to the priority queue. 


!["Heap Img"](https://raw.githubusercontent.com/mkhoshpa/mkhoshpa.github.io/master/images/indexedPQ.png?style=centerme) 


In an Indexed priority queue, we store the information in three arrays. The keys (keys), the heap representation of the priority queue (pq), and an array containing the index of each key in the heap representation (qp). So we have qp[pq[i]] = i.

We also have to modify swim and sink methods to support changing the key of an element inside the queue. 

```python
class IndexedMinPQ:
    def __init__(self,N):
        self.N = N
        self.key = [None for i in range(self.N)]
        self.pq = [None for i in range(self.N+1)]
        self.qp =[None for i in range(self.N)]
        self.total = 0

    def insert(self,i,key):
        assert type(i) == int
        if i >= self.N:
            raise IndexError('index is out of the range of IndexedMinPQ.')
        if self.key[i] is not None:
            raise IndexError('index is already in the IndexedMinPQ.')
        self.total += 1
        self.key[i] = key
        self.pq[self.total] = i
        self.qp[i] = self.total
        self.__swim(self.total)

    def __swim(self,i):
        parent_i = i//2

        while parent_i > 0 :
            key = self.key[self.pq[i]]
            parent_key = self.key[self.pq[parent_i]]
            if parent_key < key:
                break
            self.pq[i], self.pq[parent_i] = self.pq[parent_i], self.pq[i]
            self.qp[self.pq[i]] , self.qp[self.pq[parent_i]] = self.qp[self.pq[parent_i]],self.qp[self.pq[i]]
            i = parent_i
            parent_i = i // 2

    def deleteMin(self):
        if not self.isEmpty():
            out = self.key[self.pq[1]]
            self.key[self.pq[1]] = None
            self.qp[self.pq[1]] = None
            self.pq[1] = self.pq[self.total]
            self.qp[self.pq[1]] = 1
            self.pq[self.total] = None
            self.total -= 1
            self.__sink(1)
            return out
        raise IndexError('IndexedMinPQ is Empty')

    def __sink(self,i):
        child_i = i * 2
        if child_i <= self.total:
            key = self.key[self.pq[i]]
            child_key = self.key[self.pq[child_i]]
            other_child = child_i + 1
            if other_child <= self.total:
                other_child_key =  self.key[self.pq[other_child]]
                if other_child_key < child_key:
                    child_i = other_child
                    child_key = other_child_key
            if child_key < key:
                self.pq[i], self.pq[child_i] = self.pq[child_i], self.pq[i]
                self.qp[self.pq[i]], self.qp[self.pq[child_i]] = self.qp[self.pq[child_i]], self.qp[self.pq[i]]
                self.__sink(child_i)

    def isEmpty(self):
        return self.total == 0

    def decreaseKey(self,i,key):
        if i<0 or i> self.N:
            raise IndexError('index i is not in the range')
        if self.key[i] is None:
            raise IndexError('index i is not in the IndexedMinPQ')
        assert type(i) == int
        assert key < self.key[i]
        self.key[i] = key
        self.__swim(self.qp[i])

    def increaseKey(self,i,key):
        if i<0 or i> self.N:
            raise IndexError('index i is not in the range')
        if self.key[i] is None:
            raise IndexError('index i is not in the IndexedMinPQ')
        assert type(i) == int
        assert key > self.key[i]
        self.key[i] = key
        self.__sink(self.qp[i])
```
Note that even in Python, we have to know the maximum number of elements in the queue to support indexing. Since we want to associate each key with an index in order to have fast access. It's trivial to show that insert, deleting the element with the highest priority, and changing the key of a component take logarithmic time.

Reference: 
[ Algorithms, 4th Edition by Robert Sedgewick and Kevin Wayne] (https://algs4.cs.princeton.edu/lectures/keynote/43MinimumSpanningTrees.pdf)
