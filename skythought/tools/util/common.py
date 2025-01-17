import multiprocessing

class TimeoutException(Exception):
    """Custom exception for function timeout."""
    pass

def timeout(seconds):
    """Decorator to enforce a timeout on a function using multiprocessing."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # A queue to store the result or exception
            queue = multiprocessing.Queue()

            def target(queue, *args, **kwargs):
                try:
                    result = func(*args, **kwargs)
                    queue.put((True, result))
                except Exception as e:
                    queue.put((False, e))

            process = multiprocessing.Process(target=target, args=(queue, *args), kwargs=kwargs)
            process.start()
            process.join(seconds)

            if process.is_alive():
                process.terminate()
                process.join()
                raise TimeoutException(f"Function '{func.__name__}' timed out after {seconds} seconds!")

            success, value = queue.get()
            if success:
                return value
            else:
                raise value
        return wrapper
    return decorator