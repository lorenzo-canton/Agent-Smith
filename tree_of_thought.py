from ollama import Client
from pydantic import BaseModel, Field
from typing import Literal
from math import sqrt, log

self_generator = 'hermes3:8b-llama3.1-q4_K_M'
discriminator = 'granite3.1-dense:latest'

client = Client(host="http://localhost:11434")

def extract_steps(text):
    # Dividi il testo in righe
    lines = text.split('\n')
    
    # Lista per memorizzare gli step
    steps = []
    
    # Variabile per memorizzare temporaneamente le righe di uno step
    current_step = []
    
    for line in lines:
        # Salta le righe vuote
        if not line.strip():
            continue
            
        # Se la riga inizia con "Step", inizia un nuovo step
        if line.strip().startswith("Step"):
            # Se abbiamo già delle righe accumulate, aggiungiamo lo step precedente
            if current_step:
                steps.append(" ".join(current_step))
                current_step = []
            # Aggiungi il contenuto senza il prefisso "Step N:"
            content = line.split(":", 1)[1].strip()
            current_step.append(content)
        else:
            # Aggiungi la riga allo step corrente
            current_step.append(line.strip())
    
    # Aggiungi l'ultimo step
    if current_step:
        steps.append(" ".join(current_step))
    
    return steps

def extract_qa(text):
    # Dividi il testo in righe
    lines = text.split('\n')
    
    qa_pairs = []
    current_qa = {'question': '', 'answer': ''}
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('Question'):
            # Se abbiamo già una coppia Q&A, la aggiungiamo alla lista
            if current_qa['question'] and current_qa['answer']:
                qa_pairs.append(current_qa.copy())
                current_qa = {'question': '', 'answer': ''}
            
            # Estrai la domanda rimuovendo il prefisso "Question X.Y:"
            current_qa['question'] = line.split(':', 1)[1].strip()
            
        elif line.startswith('Answer'):
            # Estrai la risposta rimuovendo il prefisso "Answer X.Y:"
            current_qa['answer'] = line.split(':', 1)[1].strip()
    
    # Aggiungi l'ultima coppia Q&A
    if current_qa['question'] and current_qa['answer']:
        qa_pairs.append(current_qa)
    
    return qa_pairs

def mask_trajectory(trajectory):
    index = int(len(trajectory) * 0.2)
    return trajectory[:index + 1]

# Action 1
propose_one_step_thought = """### Instruction: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
### Response: Let's think step by step.
Step 1: Identify the initial number of trees. The problem states there are 15 trees in the grove.
Step 2: Identify the final number of trees. The problem states there will be 21 trees after the workers are done planting.
Step 3: Subtract the initial number of trees from the final number of trees to find out how many trees were planted.
Step 4: Therefore, the grove workers planted 21 (final number of trees) - 15 (initial number of trees) = 6 trees today.
Step 5: The answer is 6.

### Instruction: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
### Response: Let's think step by step.
Step 1: Start with the number of cars that are already in the parking lot, which is 3 cars.
Step 2: Add the number of cars that arrive, which is 2 cars.
Step 3: Add the numbers together. there are 3 cars + 2 cars = 5 cars in the parking lot. Step 4: The answer is 5.

### Instruction: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
### Response: Let's think step by step.
Step 1: Add the number of chocolates Leah and her sister had initially. Leah had 32 chocolates and her sister had 42 chocolates. So, they had 32 + 42 = 74 chocolates in total.
Step 2: Subtract the number of chocolates they ate from the total number of chocolates they had. They ate 35 chocolates. So, they have 74 - 35 = 39 chocolates left.
Step 3: The answer is 39.

### Instruction: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
### Response: Let's think step by step.
Step 1: Identify the initial number of lollipops. The problem states Jason had 20 lollipops.
Step 2: Identify the final number of lollipops. The problem states Jason now has 12 lollipops.
Step 3: Subtract the final number of lollipops from the initial number to find out how many lollipops Jason gave to Denny.
Step 4: Therefore, Jason gave Denny 20 (initial number of lollipops) - 12 (final number of lollipops) = 8 lollipops.
Step 5: The answer is 8.

### Instruction: {user_question}
### Response: Let's think step by step"""

# Action 2 4
answer_sub_question = """### Instruction: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
### Response: Let's think step by step. There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is: 6.

### Instruction: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
### Response: Let's think step by step. There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is: 5.

### Instruction: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
### Response: Let's think step by step. Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is: 39.

### Instruction: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
### Response: Let's think step by step. Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is: 8.

### Instruction: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
### Response: Let's think step by step. Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is: 9.

### Instruction: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
### Response: Let's think step by step. There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is: 29.

### Instruction: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
### Response: Let's think step by step. Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is: 33.

### Instruction: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
### Response: Let's think step by step. Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is: 8.

### Instruction: {user_question}
### Response:"""

class SubQuestionAnswer(BaseModel):
    response: str = Field(description="Let's think step by step.")

check_consistency = """You are a consistency checker. Your task is to evaluate whether two different answers to the same question lead to the same final numerical result, regardless of their reasoning path. You should focus only on the final numerical answers.

For each case, analyze the question and both answers, then state if they are consistent (lead to the same result) or not. 
Reply with "CONSISTENT" if both answers reach the same numerical result, or "INCONSISTENT" if they differ.

Here are some examples:

### Question: When I was 6 my sister was 3. Now I'm 56 how old is my sister?
### First answer: Let me solve this step by step. First, I'll find the age difference between you and your sister. At the given time, you were 6 and she was 3, so the difference is 3 years. Now, if you're 56, I'll subtract this difference: 56 - 3 = 53. The answer is 53.
### Second answer: The age difference between you and your sister remains constant. When you were 6, she was 3, meaning she's 3 years younger. Now that you're 56, she must be 53 years old.
CONSISTENT (Both arrive at 53)

### Question: If a train travels at 60 mph for 2.5 hours, how far does it go?
### First answer: To find the distance, I multiply speed by time: 60 mph × 2.5 hours = 150 miles.
### Second answer: Let me break this down. In one hour, the train travels 60 miles. In 2 hours, that's 120 miles. In the additional half hour (0.5 hours), it travels 30 more miles (half of 60). So in total: 120 + 30 = 150 miles.
CONSISTENT (Both arrive at 150)

### Question: I'm thinking of a number. If you multiply it by 4 and add 10, you get 26. What's the number?
### First answer: Let's solve this equation. If 4x + 10 = 26, then 4x = 16, so x = 4. The answer is 4.
### Second answer: The number is 5. When you multiply 5 by 4, you get 20. Add 10 to that and you get 30.
INCONSISTENT (First answer is 4, second answer is 5)

### Question: A rectangle has a length of 8 meters and a width of 5 meters. What is its area?
### First answer: The area is length times width: 8 × 5 = 40 square meters.
### Second answer: To find the area, I'll multiply the width by the length. A rectangle with width 5m and length 8m has an area of 5 × 8 = 40 square meters.
CONSISTENT (Both arrive at 40)

### Question: If you have $85 and spend $27 on groceries and $13 on gas, how much money do you have left?
### First answer: Starting with $85, after spending $27 and $13: 85 - 27 - 13 = 45. You have $45 left.
### Second answer: Let me calculate the total spending first: $27 + $13 = $40. Then subtract from the initial amount: $85 - $40 = $45 remains.
CONSISTENT (Both arrive at 45)

Now check if the following answers are consistent:

### Question: {user_question}
### First answer: {first_answer}
### Second answer: {second_answer}"""

class ConsistencyCheck(BaseModel):
    text_response: str = Field(description="Explain why they are consistent.")
    result: bool = Field(description="True if consistent false if not.")

class Node:
    def __init__(self, parent=None):
        self.parent: Node | None = parent
        self.child: list[Node] = []

        self.step: str | None = None
        self.visit_count: int = 0
        self.consistency_score: float = 0

    def get_root(self):
        if self.parent: return self.parent.get_root()
        return self

    def get_leaf_nodes(self):
        if len(self.child) == 0: return [self]
        leaf_nodes = []
        for child in self.child:
            leaf_nodes.extend(child.get_leaf_nodes())
        return leaf_nodes

    def get_trajectory(self):
        if self.parent: return [self, *self.parent.get_trajectory()]
        return [self]

    def update_visit_count(self):
        current = self
        while current:
            current.visit_count += 1
            current = current.parent

    def uct_score(self, exploration_constant=sqrt(2)):
        if self.visit_count == 0:
            return float('inf')
    
        # Vi/Ni: termine di sfruttamento (exploitation)
        exploitation = self.consistency_score / self.visit_count
        
        # Calcola N (visite totali del genitore)
        parent_visits = self.parent.visit_count if self.parent else 1
        
        # C * √(ln(N)/Ni): termine di esplorazione (exploration)
        exploration = exploration_constant * sqrt(log(parent_visits) / self.visit_count)
        
        return exploitation + exploration

    def select_best_child(self):
        if len(self.child) > 0: return max(self.child, key=lambda node: node.uct_score())
        return self

    def rollout_step_by_step(self):
        current_node_trajectory = self.get_trajectory()[::-1]
        current_text_trajectory = [n.step for n in current_node_trajectory if n.step]
        current_text_trajectory = current_text_trajectory[1:]
        trajectory_prompt = ""
        for i in range(len(current_text_trajectory)):
            trajectory_prompt += f"\nStep {i+1}: {current_text_trajectory[i]}"

        print(f"Rolling out trajectory for: {self.step}{trajectory_prompt}")
        text_trajectory = client.generate(
            model=self_generator,
            prompt=propose_one_step_thought.format(user_question=self.step) + trajectory_prompt
        ).response
        text_trajectory = extract_steps(text_trajectory)

        first_node = Node(self)
        first_node.step = text_trajectory[0]

        node_trajectory = [first_node]
        for step in text_trajectory[1:]: 
            new_node = Node(node_trajectory[-1])
            new_node.step = step

            node_trajectory[-1].child.append(new_node)
            node_trajectory.append(new_node)

        self.child.append(first_node)

    def complete_steps(self):
        initial_query = self.get_root().step
        current_node_trajectory = self.get_trajectory()[::-1]
        current_text_trajectory = [n.step for n in current_node_trajectory if n.step]
        masked_trajectory = mask_trajectory(current_text_trajectory)
        mask = answer_sub_question.format(user_question=initial_query) + " " + " ".join(masked_trajectory)

        question_answer = client.generate(
            model=discriminator,
            prompt=mask,
            format=SubQuestionAnswer.model_json_schema()
        )
        question_answer = SubQuestionAnswer.model_validate_json(question_answer.response)
        return question_answer.response

    def check_consistency(self, trajectory) -> ConsistencyCheck:
        initial_query = self.get_root().step
        consistency_check = client.generate(
            model=discriminator,
            prompt=check_consistency.format(user_question=initial_query, first_answer=self.step, second_answer=trajectory),
            format=ConsistencyCheck.model_json_schema()
        )
        consistency_check = ConsistencyCheck.model_validate_json(consistency_check.response)
        return consistency_check

    def expand(self, rounds=3):
        for _ in range(rounds): self.rollout_step_by_step()

    def simulate(self, rounds=3):
        print(f"Candidate solution: {self.step}")

        for round in range(rounds):
            rollout_result = self.complete_steps()

            rollout_consistency = self.check_consistency(rollout_result)
            print(f"Consistency {round}: {rollout_consistency.result}")

            if rollout_consistency.result: self.consistency_score += 1
            else: self.consistency_score -= 1


class Tree:
    def __init__(self, query):
        self.root: Node = Node()
        self.root.step = query

    def mcts_round(self):
        print("\nSelection")
        selected_node = self.root.select_best_child()
        print(f"\nSelected: {selected_node.step}")

        print("\nExpansion")
        selected_node.expand()

        print("\nSimulation")
        leaf_nodes = selected_node.get_leaf_nodes()
        for i, leaf in enumerate(leaf_nodes):
            print(f"\nCandidate Solution {i+1}:")
            leaf.simulate()

        print("\nBackpropagation")
        new_nodes = selected_node.child

        for child in new_nodes:
            child_leaf = child.get_leaf_nodes()
            for leaf in child_leaf: 
                leaf.update_visit_count()
                child.consistency_score += leaf.consistency_score
            print(f"\nStep: {child.step}")
            print(f"UCT score: {child.uct_score()}")

    def get_best_leaf(self):
        # Prendi tutte le foglie
        leaf_nodes = self.root.get_leaf_nodes()
        
        # Seleziona la foglia con il miglior rapporto consistency_score/visit_count
        best_leaf = max(leaf_nodes, 
                    key=lambda node: node.consistency_score/node.visit_count if node.visit_count > 0 else float('-inf'))
        
        return best_leaf



test_query = "When I was 6 my sister was 3. Now I'm 56 how old is my sister?"

tree = Tree(test_query)
for _ in range(3): tree.mcts_round()

best_solution = tree.get_best_leaf()
print(f"\n\nBest solution:")
print(f"Score: {best_solution.uct_score()}")
print(best_solution.step)