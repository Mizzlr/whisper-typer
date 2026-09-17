"""Exact decimal summaries for selected CSV cells."""
import re
from decimal import Decimal, InvalidOperation, localcontext


def numeric(value):
    value = value.strip()
    if re.fullmatch(r'[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?', value):
        value = value.replace(',', '')
    if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', value):
        return None
    try:
        result = Decimal(value)
        return result if result.is_finite() and abs(result.adjusted()) < 10000 else None
    except InvalidOperation:
        return None


def selection_summary(values):
    values = list(values)
    numbers = [number for value in values if (number := numeric(value)) is not None]
    result = {'cells':len(values), 'numeric':len(numbers)}
    if numbers:
        with localcontext() as ctx:
            highest = max(n.adjusted() for n in numbers)
            lowest = min(n.as_tuple().exponent for n in numbers)
            ctx.prec = max(50, highest-lowest+len(str(len(numbers)))+4)
            total = sum(numbers, Decimal(0))
            result.update(sum=total, average=total/len(numbers), minimum=min(numbers), maximum=max(numbers))
    return result


def display_summary(result):
    parts = [f"{result['cells']:,} cells", f"{result['numeric']:,} numeric"]
    for key, label in [('sum','Sum'),('average','Avg'),('minimum','Min'),('maximum','Max')]:
        if key in result:
            parts.append(f'{label} {result[key]:,.10g}')
    return '  ·  '.join(parts)
