import functools as ft

def repeat(item):
  while(True):
    yield item

def intercalate(item, list):
  return [val for pair in zip(list,repeat(item)) for val in pair]

def enum_values_str(enum) -> str:
  values : list[str] = [x.name for x in enum]
  return ft.reduce(lambda x,y: x+y,intercalate("|",values))