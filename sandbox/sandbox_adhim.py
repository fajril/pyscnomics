from dataclasses import dataclass, field

@dataclass
class Test:
    weight: int
    age: int

    def __post_init__(self):
        if self.weight > self.age:
            self.proportion = "Over"
        else:
            self.proportion = "Nope"


test1 = Test(weight=20,
             age=23)
test1.proportion = "hehehehehe"
print(test1.proportion)


