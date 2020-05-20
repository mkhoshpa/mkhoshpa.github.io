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
def sink(arr, idx):
    while 2 * idx < len(arr):
        child = 2 * idx
        if child + 1 < length:
            if arr[child + 1 ] > arr[child]:
                child = child + 1
        if arr[child] > arr[idx]:
            arr[child], arr[idx]=arr[idx], arr[child]
            idx = child
        else:
            break
            
def swim(arr,idx):
    while idx // 2 > 0:
        p = idx // 2
        if arr[idx]> arr[p]:
            arr[idx], arr[p] = arr[p], arr[idx]
            idx = p
        else:
            break
```
