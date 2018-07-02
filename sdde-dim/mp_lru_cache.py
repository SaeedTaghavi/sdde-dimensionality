from functools import namedtuple, update_wrapper, RLock

################################################################################
### LRU Cache function decorator
################################################################################

from functools import _CacheInfo, _HashedSeq, _make_key

def lru_cache(maxsize=128, typed=False, inner=None):
    """Least-recently-used cache decorator.

    This is a copy of the lru_cache in Python's functools. It differs in that it
    looks through the functions keywords for a 'pool' argument, to allow for use
    in multiprocessing: if provided, the memoized function `f` is called with
    `pool.apply_async(f, *args, **kwds)`
    Note that this prevents any 'pool' argument from being passed to the
    memoized function.
    When a 'pool' argument is provided, the returned value is an `ApplyResult`.

    If *maxsize* is set to None, the LRU features are disabled and the cache
    can grow without bound.

    If *typed* is True, arguments of different types will be cached separately.
    For example, f(3.0) and f(3) will be treated as distinct calls with
    distinct results.

    Arguments to the cached function must be hashable.

    View the cache statistics named tuple (hits, misses, maxsize, currsize)
    with f.cache_info().  Clear the cache and statistics with f.cache_clear().
    Access the underlying function with f.__wrapped__.

    See:  http://en.wikipedia.org/wiki/Cache_algorithms#Least_Recently_Used

    """

    # Users should only access the lru_cache through its public API:
    #       cache_info, cache_clear, and f.__wrapped__
    # The internals of the lru_cache are encapsulated for thread safety and
    # to allow the implementation to change (including a possible C version).

    # Early detection of an erroneous call to @lru_cache without any arguments
    # resulting in the inner function being passed to maxsize instead of an
    # integer or None.
    if maxsize is not None and not isinstance(maxsize, int):
        raise TypeError('Expected maxsize to be an integer or None')

    # Constants shared by all lru cache instances:
    sentinel = object()          # unique object used to signal cache misses
    make_key = _make_key         # build a key from the function arguments
    PREV, NEXT, KEY, RESULT = 0, 1, 2, 3   # names for the link fields

    def decorating_function(user_function):
        cache = {}
        hits = misses = 0
        full = False
        cache_get = cache.get    # bound method to lookup a key or return None
        lock = RLock()           # because linkedlist updates aren't threadsafe
        root = []                # root of the circular doubly linked list
        root[:] = [root, root, None, None]     # initialize by pointing to self

        # Added multiprocessing functionality
        nonlocal inner
        if inner is None:
            inner = user_function

        if maxsize == 0:

            def wrapper(*args, **kwds):
                # No caching -- just a statistics update after a successful call
                nonlocal misses
                # Added multiprocessing functionality
                pool = kwds.pop('pool', None)
                if pool is not None:
                    result = pool.apply_async(inner, args, kwds)
                else:
                    result = inner(*args, **kwds)
                misses += 1
                return result

        elif maxsize is None:

            def wrapper(*args, **kwds):
                # Simple caching without ordering or size limit
                nonlocal hits, misses
                pool = kwds.pop('pool', None)  # Remove before making the key
                key = make_key(args, kwds, typed)
                result = cache_get(key, sentinel)
                if result is not sentinel:
                    hits += 1
                    return result
                # Added multiprocessing functionality
                if pool is not None:
                    result = pool.apply_async(inner, args, kwds)
                else:
                    result = inner(*args, **kwds)
                cache[key] = result
                misses += 1
                return result

        else:

            def wrapper(*args, **kwds):
                # Size limited caching that tracks accesses by recency
                nonlocal root, hits, misses, full
                pool = kwds.pop('pool', None)  # Remove before making the key
                key = make_key(args, kwds, typed)
                with lock:
                    link = cache_get(key)
                    if link is not None:
                        # Move the link to the front of the circular queue
                        link_prev, link_next, _key, result = link
                        link_prev[NEXT] = link_next
                        link_next[PREV] = link_prev
                        last = root[PREV]
                        last[NEXT] = root[PREV] = link
                        link[PREV] = last
                        link[NEXT] = root
                        hits += 1
                        return result
                # Added multiprocessing functionality
                if pool is not None:
                    result = pool.apply_async(inner, args, kwds)
                else:
                    result = inner(*args, **kwds)
                with lock:
                    if key in cache:
                        # Getting here means that this same key was added to the
                        # cache while the lock was released.  Since the link
                        # update is already done, we need only return the
                        # computed result and update the count of misses.
                        pass
                    elif full:
                        # Use the old root to store the new key and result.
                        oldroot = root
                        oldroot[KEY] = key
                        oldroot[RESULT] = result
                        # Empty the oldest link and make it the new root.
                        # Keep a reference to the old key and old result to
                        # prevent their ref counts from going to zero during the
                        # update. That will prevent potentially arbitrary object
                        # clean-up code (i.e. __del__) from running while we're
                        # still adjusting the links.
                        root = oldroot[NEXT]
                        oldkey = root[KEY]
                        oldresult = root[RESULT]
                        root[KEY] = root[RESULT] = None
                        # Now update the cache dictionary.
                        del cache[oldkey]
                        # Save the potentially reentrant cache[key] assignment
                        # for last, after the root and links have been put in
                        # a consistent state.
                        cache[key] = oldroot
                    else:
                        # Put result in a new link at the front of the queue.
                        last = root[PREV]
                        link = [last, root, key, result]
                        last[NEXT] = root[PREV] = cache[key] = link
                        full = (len(cache) >= maxsize)
                    misses += 1
                return result

        def cache_info():
            """Report cache statistics"""
            with lock:
                return _CacheInfo(hits, misses, maxsize, len(cache))

        def cache_clear():
            """Clear the cache and cache statistics"""
            nonlocal hits, misses, full
            with lock:
                cache.clear()
                root[:] = [root, root, None, None]
                hits = misses = 0
                full = False

        wrapper.cache_info = cache_info
        wrapper.cache_clear = cache_clear
        wrapper.cache = cache  # DEBUG
        return update_wrapper(wrapper, user_function)

    return decorating_function
