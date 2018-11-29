

def iter_combinations(*iterables):
    if not len(iterables):
        yield tuple()
    else:
        for item in iterables[0]:
            for group in iter_combinations(*iterables[1:]):
                yield (item,) + group
