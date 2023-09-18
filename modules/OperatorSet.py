class OperatorSet:
    def __init__(self, operators):
        self.operators = operators

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
