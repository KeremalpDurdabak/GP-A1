class OperatorSet:
    def __init__(self, operators):
        self.operators = operators

    @staticmethod
    def represent(self, opNumber):
        if opNumber == 0:
            return '+'
        elif opNumber == 1:
            return '-'
        elif opNumber == 2:
            return '* 2'
        elif opNumber == 3:
            return '/ 2'
        else:
            raise ValueError(f"Out of bounds opNumber: {opNumber}")


    def compute(self, operator_select, operand1, operand2):
        operator = self.operators[operator_select]
        
        if operator == '+':
            return operand1 + operand2
        elif operator == '-':
            return operand1 - operand2
        elif operator == '*2':
            return operand1 * 2
        elif operator == '/2':
            return operand1 / 2 if operand1 != 0 else 0
        else:
            raise ValueError(f"Unknown operator: {operator}")

    def count(self):
        return len(self.operators)
