---
name: code-refactoring
description: Use when cleaning, refactoring, or improving code quality. Applies systematic refactoring techniques, identifies code smells, and ensures changes preserve functionality. Use for code cleanup, technical debt reduction, improving readability, or restructuring code.
---

# Code Refactoring Skill

Apply systematic refactoring techniques to improve code quality while preserving functionality.

## Core Principles

### The Refactoring Mindset
- **Never change behavior** - Refactoring improves structure without altering functionality
- **Small, incremental steps** - One change at a time, verify after each
- **Test coverage first** - Ensure tests exist before refactoring; run them after every change
- **Version control discipline** - Atomic commits with precise descriptions; never refactor multiple patterns simultaneously

## Code Smells to Identify

Before refactoring, identify which smells are present:

### Bloaters
- **Long Method** - Methods exceeding 40 lines or doing multiple things
- **Large Class** - Classes with too many responsibilities
- **Long Parameter List** - More than 4-5 parameters
- **Primitive Obsession** - Overuse of primitives instead of small objects
- **Data Clumps** - Groups of data that appear together repeatedly

### Object-Orientation Abusers
- **Switch Statements** - Complex conditionals that should be polymorphism
- **Temporary Field** - Fields only set in certain circumstances
- **Refused Bequest** - Subclasses not using inherited methods

### Change Preventers
- **Divergent Change** - One class changed for many different reasons
- **Shotgun Surgery** - One change requires edits across many classes
- **Parallel Inheritance Hierarchies** - Creating a subclass forces another subclass elsewhere

### Dispensables
- **Duplicate Code** - Same code structure in multiple places
- **Dead Code** - Unused variables, parameters, methods, or classes
- **Lazy Class** - Classes that don't do enough to justify existence
- **Speculative Generality** - Unused abstractions created "just in case"
- **Comments** - Comments explaining bad code instead of fixing it

### Couplers
- **Feature Envy** - Method uses another class's data more than its own
- **Inappropriate Intimacy** - Classes too dependent on each other's internals
- **Message Chains** - Long chains of method calls (a.b().c().d())
- **Middle Man** - Class that only delegates without adding value

## Refactoring Techniques

### Composing Methods

**Extract Method**
When: Method exceeds 40 lines, has duplicate logic, or is hard to test
```
# Before
def process_order(order):
    # validate
    if not order.items:
        raise ValueError("Empty order")
    if order.total < 0:
        raise ValueError("Invalid total")
    # calculate
    subtotal = sum(item.price for item in order.items)
    tax = subtotal * 0.1
    total = subtotal + tax
    # save
    db.save(order)
    return total

# After
def process_order(order):
    validate_order(order)
    total = calculate_total(order)
    save_order(order)
    return total
```

**Inline Method**
When: Method body is as clear as its name, or it just delegates without adding value

**Replace Temp with Query**
When: Temporary variables store calculations that could be methods

### Moving Features Between Objects

**Move Method**
When: A method uses another class's data more than its own (feature envy)

**Extract Class**
When: A class has multiple responsibilities that can be separated

### Organizing Data

**Encapsulate Field**
When: Public fields allow uncontrolled state modification
```
# Before
class User:
    name = ""

# After
class User:
    def __init__(self):
        self._name = ""

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value.strip()
```

**Replace Magic Number with Symbolic Constant**
When: Hard-coded numbers hide business logic
```
# Before
if age >= 21:

# After
LEGAL_DRINKING_AGE = 21
if age >= LEGAL_DRINKING_AGE:
```

### Simplifying Conditionals

**Replace Conditional with Polymorphism**
When: Complex switch/if-else statements based on type
```
# Before
def calculate_area(shape):
    if shape.type == "circle":
        return 3.14 * shape.radius ** 2
    elif shape.type == "rectangle":
        return shape.width * shape.height

# After
class Circle:
    def area(self):
        return 3.14 * self.radius ** 2

class Rectangle:
    def area(self):
        return self.width * self.height
```

**Decompose Conditional**
When: Complex conditional logic is hard to understand

**Replace Nested Conditional with Guard Clauses**
When: Deep nesting obscures the main logic path

### Simplifying Method Calls

**Introduce Parameter Object**
When: Multiple methods take the same group of parameters
```
# Before
def create_report(start_date, end_date, include_totals, format):

# After
@dataclass
class ReportOptions:
    start_date: date
    end_date: date
    include_totals: bool
    format: str

def create_report(options: ReportOptions):
```

**Replace Constructor with Factory Method**
When: Constructor contains complex logic or unclear parameters

### Dealing with Generalization

**Pull Up Method**
When: Duplicate methods exist in sibling classes

**Collapse Hierarchy**
When: Empty intermediate classes add no functionality

**Extract Interface/Superclass**
When: Multiple classes share common behavior

## Refactoring Workflow

### Before Starting
1. **Verify test coverage** - If tests don't exist, write them first
2. **Create a branch** - Isolate refactoring work
3. **Identify the smell** - Name what you're fixing
4. **Plan the approach** - Choose appropriate technique(s)

### During Refactoring
1. **One change at a time** - Never combine multiple refactorings
2. **Run tests after each change** - Catch regressions immediately
3. **Commit frequently** - Small, atomic commits with clear messages
4. **Keep changes reversible** - Enable easy rollback

### After Refactoring
1. **Run full test suite** - Verify nothing broke
2. **Review the diff** - Ensure only intended changes
3. **Check for new smells** - Refactoring can reveal other issues
4. **Document if needed** - Update comments/docs for significant changes

## When NOT to Refactor

- **No test coverage** - Write tests first
- **Tight deadline** - Schedule refactoring for later
- **Unclear requirements** - Wait until behavior is well-defined
- **Complete rewrite needed** - Sometimes it's better to start fresh

## Clean Code Principles

### Naming
- **Reveal intent** - Names should explain purpose without comments
- **Avoid abbreviations** - `calculateTotalPrice()` not `calcTotPrc()`
- **Use domain language** - Match business terminology

### Functions
- **Single responsibility** - Do one thing well
- **Small size** - Aim for under 20 lines
- **Few parameters** - Ideally 0-2, max 3-4
- **No side effects** - Or clearly document them

### Comments
- **Prefer self-documenting code** - Rename instead of commenting
- **Explain why, not what** - Code shows what, comments explain why
- **Keep updated** - Outdated comments are worse than none

### DRY (Don't Repeat Yourself)
- **Extract shared logic** - But only when there's true duplication
- **Avoid premature abstraction** - Wait for 3+ occurrences
- **Consider if changes should propagate** - Not all similar code is duplicate

## Sources

- [12 Essential Code Refactoring Techniques - Augment Code](https://www.augmentcode.com/guides/12-essential-code-refactoring-techniques)
- [Refactoring Guru](https://refactoring.guru/refactoring)
- [Writing Clean Code - Datanovia](https://www.datanovia.com/learn/programming/best-practices/writing-clean-code.html)
- [Code Refactoring Best Practices - Mad Devs](https://maddevs.io/blog/code-refactoring/)
