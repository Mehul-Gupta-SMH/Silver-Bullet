"""
generate_data.py
----------------
Generates train / validate / test JSON files in data/
Format: {"data": [{"text1": ..., "text2": ..., "label": 0|1}]}

Pair taxonomy
  label=1  positive      – semantically equivalent / faithful
  label=0  hard_neg      – same topic/structure but factually wrong, negated, or partially unfaithful
  label=0  soft_neg      – clearly different topic or domain

Split: 70 / 15 / 15  (train / validate / test)
"""

import json, os, random
random.seed(42)

# ---------------------------------------------------------------------------
# Data — (text1, text2, label, kind)
# ---------------------------------------------------------------------------
PAIRS = [

# ════════════════════════════════════════════════════════════════════════════
# POSITIVES  (label = 1)
# ════════════════════════════════════════════════════════════════════════════

# ── Physics ──────────────────────────────────────────────────────────────────
("Newton's first law of motion states that an object at rest stays at rest, and an object in motion continues in motion at constant velocity unless acted upon by a net external force.",
 "According to Newton's first law, any object maintains its current state of motion — stationary or moving at constant speed — unless a net external force acts on it.",
 1, "positive"),

("The speed of light in a vacuum is approximately 299,792,458 metres per second.",
 "Light travels through a vacuum at roughly 299,792,458 m/s, commonly approximated as 3 × 10⁸ m/s.",
 1, "positive"),

("Einstein's mass-energy equivalence formula E = mc² states that energy equals mass multiplied by the square of the speed of light.",
 "The equation E = mc² expresses that mass and energy are interconvertible: a small amount of mass corresponds to an enormous amount of energy, scaled by c².",
 1, "positive"),

("Gravity pulls objects toward each other with a force proportional to their masses and inversely proportional to the square of the distance between them.",
 "The gravitational force between two bodies increases with their masses and decreases as the square of the separation between them grows — this is Newton's law of universal gravitation.",
 1, "positive"),

("Thermodynamics' second law states that the total entropy of an isolated system can never decrease over time.",
 "According to the second law of thermodynamics, entropy in a closed system always tends to increase or remain constant — it cannot spontaneously decrease.",
 1, "positive"),

("Sound travels faster in denser media; it moves at approximately 343 m/s in air and about 1,480 m/s in water at room temperature.",
 "The speed of sound depends on the medium: roughly 343 m/s through air and around 1,480 m/s through water under standard conditions.",
 1, "positive"),

# ── Biology ──────────────────────────────────────────────────────────────────
("Photosynthesis is the process by which plants convert light energy into chemical energy stored as glucose, using carbon dioxide and water as inputs and releasing oxygen as a by-product.",
 "Through photosynthesis, plants absorb CO₂ and water and use sunlight to produce glucose (chemical energy), releasing oxygen in the process.",
 1, "positive"),

("DNA is a double-helix molecule composed of nucleotides, with adenine pairing with thymine and guanine pairing with cytosine.",
 "The DNA molecule has a double-helix structure; its base-pairing rules are A–T and G–C.",
 1, "positive"),

("Mitosis is cell division that produces two genetically identical daughter cells, each with the same number of chromosomes as the parent cell.",
 "During mitosis, a single cell divides to form two daughter cells that are genetically identical to the original and contain the same chromosome count.",
 1, "positive"),

("The human body contains approximately 37 trillion cells, each performing specialised functions to maintain life.",
 "About 37 trillion cells make up the human body, each carrying out specific roles that collectively sustain biological functions.",
 1, "positive"),

("Evolution by natural selection is the process by which heritable traits that increase survival and reproduction become more common in a population over generations.",
 "Natural selection drives evolution: traits that improve an organism's ability to survive and reproduce spread through the population across generations.",
 1, "positive"),

("CRISPR-Cas9 is a gene-editing tool that allows scientists to precisely modify DNA sequences in living organisms.",
 "CRISPR-Cas9 enables precise, targeted edits to the DNA of living organisms, making it a powerful tool for genetic engineering.",
 1, "positive"),

("The mitochondria are often called the powerhouse of the cell because they produce ATP through cellular respiration.",
 "Mitochondria generate ATP — the cell's primary energy currency — via cellular respiration, earning them the nickname 'powerhouse of the cell'.",
 1, "positive"),

("mRNA vaccines work by delivering messenger RNA into cells, instructing them to produce a protein that triggers an immune response.",
 "In mRNA vaccination, synthetic messenger RNA is introduced into the body's cells; the cells read it and produce an antigen that primes the immune system.",
 1, "positive"),

# ── Chemistry ────────────────────────────────────────────────────────────────
("Water is a polar molecule with the chemical formula H₂O, consisting of two hydrogen atoms covalently bonded to one oxygen atom.",
 "H₂O — water — is a polar covalent molecule formed by two hydrogen atoms bonded to a single oxygen atom.",
 1, "positive"),

("The pH scale measures the acidity or alkalinity of a solution on a logarithmic scale from 0 to 14, with 7 being neutral.",
 "pH is a logarithmic measure of hydrogen ion concentration: 0–6 is acidic, 7 is neutral, and 8–14 is alkaline.",
 1, "positive"),

("Oxidation involves the loss of electrons, while reduction involves the gain of electrons — together they form redox reactions.",
 "In redox chemistry, oxidation is the loss of electrons and reduction is the gain of electrons; the two always occur together.",
 1, "positive"),

# ── Mathematics ──────────────────────────────────────────────────────────────
("The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of the squares of the other two sides: a² + b² = c².",
 "For any right-angled triangle, a² + b² = c², where c is the hypotenuse — this is the Pythagorean theorem.",
 1, "positive"),

("A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself.",
 "Prime numbers are integers greater than 1 whose only factors are 1 and the number itself.",
 1, "positive"),

("The derivative of a function measures the rate of change of the function's output with respect to its input.",
 "In calculus, a derivative represents how rapidly a function's value changes as its input varies.",
 1, "positive"),

("Euler's identity, e^(iπ) + 1 = 0, is widely regarded as the most beautiful equation in mathematics because it links five fundamental constants.",
 "e^(iπ) + 1 = 0, known as Euler's identity, elegantly connects e, i, π, 1, and 0 — five of mathematics' most important constants.",
 1, "positive"),

# ── Computer Science ─────────────────────────────────────────────────────────
("A binary search algorithm finds a target value in a sorted array by repeatedly halving the search interval, achieving O(log n) time complexity.",
 "Binary search narrows down a sorted list by comparing the midpoint to the target and eliminating half the remaining elements each step, running in O(log n) time.",
 1, "positive"),

("A hash table stores key-value pairs and provides average O(1) time complexity for insertions, deletions, and lookups using a hash function.",
 "Hash maps use a hash function to map keys to bucket indices, enabling constant-time average-case insertions, deletions, and lookups.",
 1, "positive"),

("Git is a distributed version control system that tracks changes in source code and allows multiple developers to collaborate on a project.",
 "Git provides distributed version control, enabling developers to track code changes, branch independently, and merge contributions from multiple contributors.",
 1, "positive"),

("REST APIs communicate over HTTP using standard methods — GET, POST, PUT, DELETE — and return data typically in JSON or XML format.",
 "A RESTful API uses HTTP verbs (GET, POST, PUT, DELETE) to expose resources and exchange data, commonly serialised as JSON.",
 1, "positive"),

("A neural network is a computational model inspired by the brain, consisting of layers of interconnected nodes (neurons) that learn patterns from data.",
 "Neural networks are machine-learning models made up of layers of artificial neurons. They learn by adjusting weights based on training data.",
 1, "positive"),

("Recursion is a programming technique where a function calls itself to solve smaller instances of the same problem until a base case is reached.",
 "A recursive function solves a problem by breaking it into smaller sub-problems of the same type and calling itself until it hits a stopping condition (base case).",
 1, "positive"),

("SQL's JOIN clause combines rows from two or more tables based on a related column, enabling relational queries across multiple datasets.",
 "In SQL, JOIN merges records from multiple tables using a shared column, allowing queries to retrieve related data from different tables simultaneously.",
 1, "positive"),

("Docker containers package an application and its dependencies together, ensuring it runs consistently across different environments.",
 "A Docker container bundles code with all its libraries and configuration so the same image runs identically on any host machine.",
 1, "positive"),

("Big-O notation describes the upper-bound worst-case time or space complexity of an algorithm as a function of input size.",
 "Big-O expresses how an algorithm's resource usage scales in the worst case as the input grows larger.",
 1, "positive"),

("A linked list is a data structure where each element (node) contains a value and a pointer to the next node in the sequence.",
 "In a linked list, each node stores data plus a reference to the next node, forming a chain rather than a contiguous memory block.",
 1, "positive"),

# ── Medicine ─────────────────────────────────────────────────────────────────
("Metformin is a first-line treatment for type 2 diabetes that works by decreasing hepatic glucose production and improving insulin sensitivity.",
 "As the primary drug for type 2 diabetes management, metformin reduces the liver's glucose output and helps cells respond more effectively to insulin.",
 1, "positive"),

("Antibiotics kill bacteria or inhibit their growth; they are ineffective against viral infections such as the common cold or influenza.",
 "Antibiotic drugs target bacterial cells and are useless against viruses — colds and flu, caused by viruses, do not respond to antibiotic treatment.",
 1, "positive"),

("Blood pressure is recorded as two values: systolic pressure (when the heart beats) and diastolic pressure (when the heart rests between beats).",
 "A blood pressure reading consists of the systolic value (pressure during heartbeats) over the diastolic value (pressure between beats).",
 1, "positive"),

("The placebo effect is when a patient experiences a real improvement in symptoms after receiving an inactive treatment, due to the belief that it will work.",
 "Placebo effect describes genuine symptom improvement following an inert treatment, driven by the patient's expectation of benefit.",
 1, "positive"),

("Statins are medications that lower LDL cholesterol by inhibiting the HMG-CoA reductase enzyme responsible for cholesterol synthesis in the liver.",
 "By blocking HMG-CoA reductase — the liver enzyme that makes cholesterol — statins reduce LDL ('bad') cholesterol levels in the blood.",
 1, "positive"),

("Vaccines stimulate the immune system to recognise and destroy a specific pathogen by exposing it to a weakened or inactivated form of the disease agent.",
 "Vaccination trains the immune system to fight a pathogen by introducing a harmless version (weakened or inactivated) that triggers antibody production without causing disease.",
 1, "positive"),

# ── History ──────────────────────────────────────────────────────────────────
("The French Revolution began in 1789 and transformed France from an absolute monarchy into a republic, culminating in the execution of Louis XVI in 1793.",
 "Starting in 1789, the French Revolution overthrew the absolute monarchy, established a republic, and led to King Louis XVI being executed in 1793.",
 1, "positive"),

("World War II lasted from 1939 to 1945, involved most of the world's nations, and resulted in approximately 70-85 million deaths.",
 "The Second World War, fought from 1939 to 1945, drew in nations from across the globe and caused an estimated 70 to 85 million fatalities.",
 1, "positive"),

("The Berlin Wall, built in 1961 to divide East and West Berlin, fell on 9 November 1989 as a symbol of the Cold War's end.",
 "Erected in 1961 to separate Communist East Berlin from West Berlin, the Berlin Wall came down on 9 November 1989, marking the effective end of the Cold War.",
 1, "positive"),

("The Apollo 11 mission successfully landed astronauts Neil Armstrong and Buzz Aldrin on the Moon on 20 July 1969, with Armstrong becoming the first human to walk on the lunar surface.",
 "On 20 July 1969, Apollo 11 touched down on the Moon; Neil Armstrong stepped onto the surface first, followed by Buzz Aldrin — making history as the first crewed lunar landing.",
 1, "positive"),

("The Industrial Revolution, which began in Britain in the late 18th century, marked the transition from agrarian economies to manufacturing-based ones, powered by steam and coal.",
 "Beginning in late-18th-century Britain, the Industrial Revolution shifted economies from farming to factory production, driven primarily by steam engines and coal.",
 1, "positive"),

# ── Economics / Finance ───────────────────────────────────────────────────────
("Inflation is the rate at which the general price level of goods and services rises over time, eroding purchasing power.",
 "When inflation rises, the same amount of money buys fewer goods and services — it represents a sustained increase in the overall price level.",
 1, "positive"),

("Compound interest is interest calculated on both the principal and the accumulated interest from previous periods.",
 "Unlike simple interest, compound interest grows on the original principal plus any interest already earned, leading to exponential growth over time.",
 1, "positive"),

("Supply and demand determines market price: when supply exceeds demand, prices fall; when demand exceeds supply, prices rise.",
 "Market prices are set by the interaction of supply and demand — surplus supply drives prices down, while excess demand pushes them up.",
 1, "positive"),

("Gross domestic product (GDP) is the total monetary value of all goods and services produced within a country's borders in a given period.",
 "GDP measures the aggregate economic output of a nation — the total value of finished goods and services produced domestically in a specific timeframe.",
 1, "positive"),

# ── Code Pairs ───────────────────────────────────────────────────────────────
("def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n - 1)",
 "def factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result",
 1, "positive"),

("def is_palindrome(s):\n    return s == s[::-1]",
 "def is_palindrome(s):\n    left, right = 0, len(s) - 1\n    while left < right:\n        if s[left] != s[right]:\n            return False\n        left += 1; right -= 1\n    return True",
 1, "positive"),

("SELECT name, salary FROM employees WHERE department = 'Engineering' ORDER BY salary DESC;",
 "SELECT name, salary FROM employees WHERE department = 'Engineering' ORDER BY salary DESC LIMIT 100;",
 1, "positive"),

("# Python list comprehension\nsquares = [x**2 for x in range(10)]",
 "# Equivalent using map\nsquares = list(map(lambda x: x**2, range(10)))",
 1, "positive"),

# ── Environment & Climate ─────────────────────────────────────────────────────
("Global average temperatures have risen approximately 1.1 °C above pre-industrial levels as of 2023, driven primarily by human greenhouse gas emissions.",
 "As of 2023, the Earth is roughly 1.1 degrees Celsius warmer than it was before industrialisation, largely due to CO₂ and other greenhouse gases released by human activity.",
 1, "positive"),

("Renewable energy sources such as solar and wind do not emit greenhouse gases during operation and are key to reducing carbon emissions.",
 "Solar and wind power generate electricity without emitting CO₂ during operation, making them essential tools in the transition to a low-carbon energy system.",
 1, "positive"),

("Deforestation removes trees that absorb CO₂, contributing to climate change while also destroying habitats and reducing biodiversity.",
 "When forests are cleared, carbon-absorbing trees are lost, which accelerates climate change; deforestation also destroys ecosystems and threatens biodiversity.",
 1, "positive"),

# ── Psychology ────────────────────────────────────────────────────────────────
("Cognitive dissonance is the mental discomfort experienced when a person holds two or more contradictory beliefs, values, or ideas simultaneously.",
 "When someone holds conflicting beliefs at the same time, the resulting psychological tension is called cognitive dissonance.",
 1, "positive"),

("The confirmation bias is the tendency to search for, interpret, and remember information in a way that confirms one's pre-existing beliefs.",
 "Confirmation bias leads people to favour information that supports what they already believe and to discount evidence that contradicts it.",
 1, "positive"),

("Operant conditioning is a learning process where behaviour is shaped by rewards (positive reinforcement) and punishments.",
 "In operant conditioning, behaviours increase when followed by rewards and decrease when followed by punishments — a key concept in behavioural psychology.",
 1, "positive"),

# ── Literature ────────────────────────────────────────────────────────────────
("In George Orwell's 1984, the totalitarian Party led by Big Brother uses constant surveillance, propaganda, and psychological manipulation to control citizens.",
 "Orwell's 1984 portrays a dystopian state where the all-powerful Party maintains control through pervasive surveillance, propaganda, and psychological coercion.",
 1, "positive"),

("Shakespeare's Hamlet explores themes of revenge, mortality, indecision, and the corruption of power through the story of a Danish prince.",
 "Hamlet by Shakespeare follows a Danish prince navigating revenge, existential doubt, and the moral corruption of those around him.",
 1, "positive"),

# ── Nutrition / Lifestyle ─────────────────────────────────────────────────────
("Regular aerobic exercise for at least 150 minutes per week significantly reduces the risk of cardiovascular disease, type 2 diabetes, and depression.",
 "Getting at least 150 minutes of moderate aerobic activity weekly substantially lowers the risk of heart disease, diabetes, and mental health disorders.",
 1, "positive"),

("A Mediterranean diet, rich in vegetables, whole grains, olive oil, and fish, is associated with reduced risk of heart disease and longer lifespan.",
 "Research links the Mediterranean diet — high in vegetables, whole grains, fish, and olive oil — with lower cardiovascular risk and improved longevity.",
 1, "positive"),

# ── AI / ML ──────────────────────────────────────────────────────────────────
("Transformer models use self-attention mechanisms to process sequences in parallel, enabling them to capture long-range dependencies more efficiently than RNNs.",
 "Unlike RNNs, Transformers process entire sequences simultaneously through self-attention, making them better at modelling relationships between distant tokens.",
 1, "positive"),

("Overfitting occurs when a machine learning model learns the training data too well, including noise, and performs poorly on unseen data.",
 "A model that overfits has memorised the training set — including its noise — and fails to generalise to new, unseen examples.",
 1, "positive"),

("Gradient descent is an optimisation algorithm that iteratively updates model parameters in the direction that minimises the loss function.",
 "In gradient descent, model weights are repeatedly adjusted in the opposite direction of the gradient of the loss, gradually reducing the error.",
 1, "positive"),

("Reinforcement learning trains an agent to make decisions by rewarding desired behaviours and penalising undesired ones through trial and error.",
 "An RL agent learns an optimal policy by interacting with its environment — earning rewards for good actions and penalties for bad ones.",
 1, "positive"),

("Cross-entropy loss measures the difference between a predicted probability distribution and the true distribution, commonly used for classification tasks.",
 "For classification, cross-entropy loss quantifies how far the model's predicted probabilities diverge from the actual one-hot labels.",
 1, "positive"),

# ── Geography ─────────────────────────────────────────────────────────────────
("The Amazon River carries roughly 20% of the world's fresh water discharge into the ocean and is the largest river by water volume.",
 "By volume, the Amazon is the world's largest river, responsible for about one-fifth of all freshwater discharged into the oceans globally.",
 1, "positive"),

("Mount Everest, located in the Himalayas on the border of Nepal and Tibet, is the highest peak on Earth at 8,848.86 metres above sea level.",
 "At 8,848.86 m, Mount Everest in the Nepal–Tibet Himalayas is the tallest mountain on Earth measured from sea level.",
 1, "positive"),

# ── Law ───────────────────────────────────────────────────────────────────────
("The principle of innocent until proven guilty places the burden of proof on the prosecution, not the defendant, in criminal trials.",
 "In criminal law, defendants are presumed innocent; it is the prosecution's responsibility to prove guilt beyond reasonable doubt.",
 1, "positive"),

("GDPR requires organisations to obtain explicit consent before collecting personal data and gives individuals the right to access, correct, or delete their data.",
 "Under GDPR, companies must get clear consent before gathering personal data, and individuals retain the right to access, rectify, or erase their information.",
 1, "positive"),

# ── Space ─────────────────────────────────────────────────────────────────────
("The James Webb Space Telescope, launched in December 2021, observes the universe in infrared light and can detect galaxies formed shortly after the Big Bang.",
 "Launched in December 2021, JWST uses infrared imaging to peer at the earliest galaxies, offering views of the universe from just a few hundred million years after the Big Bang.",
 1, "positive"),

("Black holes are regions of spacetime where gravity is so strong that nothing — not even light — can escape once it crosses the event horizon.",
 "A black hole's gravity is intense enough to trap everything within its event horizon, including light, making it invisible by definition.",
 1, "positive"),

# ── Misc Paraphrases ─────────────────────────────────────────────────────────
("The Great Wall of China stretches over 21,000 kilometres and was built to protect Chinese states from nomadic invasions.",
 "Spanning more than 21,000 km, the Great Wall was constructed over centuries to defend Chinese territories against northern nomadic incursions.",
 1, "positive"),

("Shakespeare was born in Stratford-upon-Avon in 1564 and is widely considered the greatest playwright in the English language.",
 "Born in Stratford-upon-Avon in 1564, William Shakespeare is broadly regarded as the most important writer in the English literary tradition.",
 1, "positive"),

("The Eiffel Tower was built between 1887 and 1889 as the entrance arch for the 1889 World's Fair, designed by Gustave Eiffel.",
 "Designed by Gustave Eiffel and constructed from 1887 to 1889, the Eiffel Tower was originally the grand entrance gate for the Paris World's Fair of 1889.",
 1, "positive"),

("Vaccines have eradicated smallpox and dramatically reduced the incidence of polio, measles, and other infectious diseases worldwide.",
 "The global vaccination effort has completely eliminated smallpox and brought polio, measles, and other infectious diseases to historic lows.",
 1, "positive"),

("The internet is a global system of interconnected computer networks that enables communication, commerce, and access to information for billions of people.",
 "Billions of people use the internet — a worldwide network of interconnected computers — to communicate, conduct business, and access information.",
 1, "positive"),

("Machine learning is a branch of artificial intelligence in which systems learn from data to improve their performance on tasks without being explicitly programmed.",
 "ML is a subset of AI where algorithms learn patterns from data and improve over time without requiring hand-coded rules for every scenario.",
 1, "positive"),

("Climate change refers to long-term shifts in global temperatures and weather patterns, primarily caused by human activities since the Industrial Revolution.",
 "Long-term changes in temperature and weather patterns — driven mainly by industrialisation and fossil fuel use — constitute what we call climate change.",
 1, "positive"),

("The stock market allows companies to raise capital by selling shares to public investors, who then own a fraction of the company.",
 "By issuing shares on the stock market, companies access public capital, and shareholders receive partial ownership of the business in return.",
 1, "positive"),

("Sleep is essential for memory consolidation, immune function, and emotional regulation; adults generally need 7–9 hours per night.",
 "Adults typically need 7 to 9 hours of sleep nightly to support memory formation, immune health, and emotional well-being.",
 1, "positive"),

("The human immune system consists of two main components: the innate immune system, which provides immediate non-specific defence, and the adaptive immune system, which mounts a targeted response.",
 "Immunity relies on two systems: the innate immune system offers rapid, generalised protection, while the adaptive immune system produces specific responses to recognised pathogens.",
 1, "positive"),

("Blockchain is a distributed ledger technology that records transactions in a chain of cryptographically linked blocks, making it tamper-resistant.",
 "A blockchain stores data as a chain of cryptographically secured blocks across a distributed network, making historic records extremely difficult to alter.",
 1, "positive"),

("Abstract classes in object-oriented programming define a template for subclasses but cannot be instantiated directly.",
 "In OOP, an abstract class provides a blueprint that child classes must implement; it cannot be used to create objects directly.",
 1, "positive"),

("The central limit theorem states that the distribution of sample means approaches a normal distribution as sample size increases, regardless of the population's distribution.",
 "As sample size grows, the CLT guarantees that the sampling distribution of the mean converges to a normal distribution, no matter what the underlying population looks like.",
 1, "positive"),

("Osmosis is the movement of water molecules across a semi-permeable membrane from a region of lower solute concentration to a region of higher solute concentration.",
 "Water moves by osmosis from a dilute solution to a more concentrated one through a semi-permeable membrane, equalising solute concentrations on both sides.",
 1, "positive"),

("The First Amendment to the US Constitution protects freedom of speech, religion, press, peaceful assembly, and the right to petition the government.",
 "Under the First Amendment, Americans have constitutional protection for free speech, religious freedom, freedom of the press, peaceful protest, and the right to petition.",
 1, "positive"),

("Object-oriented programming organises software around objects — data structures that combine state (attributes) and behaviour (methods).",
 "OOP models software as collections of objects, each encapsulating its own data (fields) and operations (methods) that act on that data.",
 1, "positive"),

("The law of conservation of energy states that energy cannot be created or destroyed, only converted from one form to another.",
 "Energy is neither created nor destroyed — it only transforms between forms such as kinetic, potential, thermal, or chemical energy.",
 1, "positive"),

("Bayes' theorem describes the probability of an event based on prior knowledge of conditions related to the event: P(A|B) = P(B|A)·P(A)/P(B).",
 "Bayes' theorem updates the probability of a hypothesis given new evidence: P(A|B) = P(B|A) × P(A) / P(B).",
 1, "positive"),

("Antibody tests detect the presence of antibodies in the blood, indicating past exposure or immune response to a pathogen or vaccine.",
 "Serological (antibody) tests look for immune proteins in the bloodstream that signal prior infection or vaccination.",
 1, "positive"),

("TCP/IP is the foundational protocol suite of the internet, governing how data is packaged, addressed, transmitted, routed, and received.",
 "The internet runs on TCP/IP — a set of rules that defines how data is broken into packets, addressed, transmitted across networks, and reassembled at the destination.",
 1, "positive"),

("In Python, list comprehensions provide a concise way to create lists: [expression for item in iterable if condition].",
 "Python list comprehensions let you build lists in a single line: [expr for elem in sequence if predicate], replacing verbose for-loop constructs.",
 1, "positive"),

("The greenhouse effect occurs when gases like CO₂, methane, and water vapour trap heat in the atmosphere, warming the Earth's surface.",
 "Greenhouse gases — CO₂, methane, water vapour — absorb outgoing infrared radiation and re-radiate it back toward the surface, raising global temperatures.",
 1, "positive"),

("Opportunity cost is the value of the next best alternative foregone when making a decision.",
 "Every choice carries an opportunity cost — the benefit you give up by not choosing the next best option.",
 1, "positive"),

("Antibiotics resistance arises when bacteria evolve mechanisms to survive exposure to drugs that previously killed them.",
 "When bacteria develop the ability to withstand antibiotic treatment through mutation or gene transfer, antibiotic resistance results.",
 1, "positive"),

("A convolution neural network (CNN) uses learnable filters to detect local patterns such as edges and textures in images, enabling spatial feature extraction.",
 "CNNs apply trainable convolutional filters to input images to identify spatial features like edges, corners, and textures at various scales.",
 1, "positive"),

("Public key cryptography uses a pair of mathematically linked keys — a public key to encrypt and a private key to decrypt — enabling secure communication without sharing a secret.",
 "In asymmetric cryptography, a public key encrypts data while only the paired private key can decrypt it, allowing secure exchange without a pre-shared secret.",
 1, "positive"),

("The peer-review process requires that scientific findings be evaluated by other experts in the field before publication, ensuring quality and validity.",
 "Before a scientific paper is published, peer review subjects it to scrutiny by independent experts who assess its methodology, results, and conclusions.",
 1, "positive"),

("Homeostasis is the body's ability to maintain stable internal conditions (temperature, pH, blood sugar) despite changes in the external environment.",
 "Through homeostasis, living organisms regulate internal variables — body temperature, blood pH, glucose levels — within narrow ranges regardless of external changes.",
 1, "positive"),

("The placebo-controlled randomised trial is the gold standard for evaluating the effectiveness of medical interventions.",
 "Randomised controlled trials with placebo comparison groups are the highest-quality evidence for determining whether a medical treatment works.",
 1, "positive"),

("Machine translation systems like Google Translate use neural networks trained on large bilingual corpora to convert text from one language to another.",
 "Tools such as Google Translate employ neural network models, trained on millions of paired translated documents, to automatically render text in a target language.",
 1, "positive"),

("Dark matter makes up approximately 27% of the universe's total energy content but cannot be directly observed — it only reveals itself through gravitational effects.",
 "About 27% of the cosmos consists of dark matter, an unseen substance detectable only by its gravitational influence on visible matter and light.",
 1, "positive"),

("The SOLID principles are five object-oriented design guidelines intended to make software more maintainable, flexible, and scalable.",
 "SOLID is an acronym for five OOP design principles — Single responsibility, Open/closed, Liskov substitution, Interface segregation, Dependency inversion — that promote clean, maintainable code.",
 1, "positive"),

("Recursion without a proper base case leads to infinite recursion and a stack overflow error.",
 "If a recursive function never reaches its base case, it calls itself indefinitely until the call stack overflows.",
 1, "positive"),

# ════════════════════════════════════════════════════════════════════════════
# HARD NEGATIVES  (label = 0) — same topic / structure, but factually wrong,
#                              negated, or partially unfaithful
# ════════════════════════════════════════════════════════════════════════════

# ── Wrong numbers / dates ────────────────────────────────────────────────────
("Newton's first law of motion states that an object at rest stays at rest, and an object in motion continues in motion at constant velocity unless acted upon by a net external force.",
 "Newton's second law of motion states that force equals mass times acceleration (F = ma), and an object at rest stays at rest unless acted upon by a net external force.",
 0, "hard_neg"),

("The speed of light in a vacuum is approximately 299,792,458 metres per second.",
 "The speed of light in a vacuum is approximately 300,000 kilometres per hour, or about 186 miles per second.",
 0, "hard_neg"),

("World War II lasted from 1939 to 1945, involved most of the world's nations, and resulted in approximately 70-85 million deaths.",
 "World War II lasted from 1941 to 1945 — after the US entered the conflict — and resulted in approximately 40 million deaths, most of them military personnel.",
 0, "hard_neg"),

("The Berlin Wall fell on 9 November 1989 as a symbol of the Cold War's end.",
 "The Berlin Wall was demolished in 1991 following the formal dissolution of the Soviet Union, marking the official end of the Cold War.",
 0, "hard_neg"),

("Global average temperatures have risen approximately 1.1 °C above pre-industrial levels as of 2023.",
 "Global average temperatures have risen approximately 2.5 °C above pre-industrial levels as of 2023, already exceeding the Paris Agreement target.",
 0, "hard_neg"),

("Mount Everest is the highest peak on Earth at 8,848.86 metres above sea level.",
 "Mount Everest stands at 8,611 metres above sea level, making it the world's tallest mountain — slightly shorter than K2 which peaks at 8,848 metres.",
 0, "hard_neg"),

("The Apollo 11 mission landed astronauts on the Moon on 20 July 1969.",
 "The Apollo 13 mission successfully landed astronauts on the Moon in April 1970, completing the second crewed lunar landing.",
 0, "hard_neg"),

("The French Revolution began in 1789 and culminated in the execution of Louis XVI in 1793.",
 "The French Revolution began in 1799 with Napoleon's coup and culminated in the execution of Louis XVI, who had already died in 1793.",
 0, "hard_neg"),

# ── Negation / opposite meaning ──────────────────────────────────────────────
("Antibiotics kill bacteria or inhibit their growth; they are ineffective against viral infections such as the common cold.",
 "Antibiotics are highly effective against viral infections such as the common cold and flu, and are routinely prescribed for these illnesses.",
 0, "hard_neg"),

("Metformin works by decreasing hepatic glucose production and improving insulin sensitivity. It does NOT stimulate insulin secretion.",
 "Metformin works primarily by stimulating the pancreas to secrete more insulin, which lowers blood glucose levels in type 2 diabetes patients.",
 0, "hard_neg"),

("The law of conservation of energy states that energy cannot be created or destroyed.",
 "The law of energy conservation states that energy can be created in nuclear reactions and destroyed in chemical reactions, but not in mechanical systems.",
 0, "hard_neg"),

("DNA is a double-helix molecule with adenine pairing with thymine and guanine pairing with cytosine.",
 "DNA is a double-helix molecule where adenine pairs with cytosine and guanine pairs with thymine — held together by hydrogen bonds.",
 0, "hard_neg"),

("The greenhouse effect warms the Earth by trapping heat in the atmosphere.",
 "The greenhouse effect cools the Earth by reflecting solar radiation back into space before it can warm the surface.",
 0, "hard_neg"),

("Gradient descent updates model parameters in the direction that minimises the loss function.",
 "Gradient ascent updates model parameters in the direction that maximises the loss function, commonly used in generative adversarial networks for the discriminator.",
 0, "hard_neg"),

("Vaccines stimulate the immune system by exposing it to a weakened or inactivated form of the pathogen.",
 "Vaccines work by introducing live, fully active pathogens into the body, which then directly infect cells to trigger immunity.",
 0, "hard_neg"),

("The principle of innocent until proven guilty places the burden of proof on the prosecution.",
 "In criminal trials, the principle of innocent until proven guilty places the burden of proof on the defendant to demonstrate their innocence.",
 0, "hard_neg"),

# ── Same structure, swapped entities ─────────────────────────────────────────
("Shakespeare was born in Stratford-upon-Avon in 1564.",
 "Shakespeare was born in London in 1564 and later moved to Stratford-upon-Avon, where he retired and died.",
 0, "hard_neg"),

("The Eiffel Tower was designed by Gustave Eiffel and built between 1887 and 1889.",
 "The Eiffel Tower was designed by architect Henri Labrouste and built between 1892 and 1894 for the Paris Olympics.",
 0, "hard_neg"),

("Isaac Newton developed the laws of motion and universal gravitation in the 17th century.",
 "Albert Einstein developed the laws of motion and universal gravitation in the early 20th century, replacing Newton's earlier theories.",
 0, "hard_neg"),

# ── Partial faithfulness with added hallucination ─────────────────────────────
("Photosynthesis uses light, water, and CO₂ to produce glucose and releases oxygen.",
 "Photosynthesis uses light, water, and CO₂ to produce glucose and releases oxygen. It takes place primarily in the roots of plants, where chlorophyll is most concentrated.",
 0, "hard_neg"),

("Metformin is a first-line treatment for type 2 diabetes. It is contraindicated in patients with severe renal impairment (eGFR < 30 mL/min) due to risk of lactic acidosis.",
 "Metformin is a first-line treatment for type 2 diabetes. It is generally safe for patients with kidney disease of any severity and is also effective for type 1 diabetes.",
 0, "hard_neg"),

("The Amazon River carries roughly 20% of the world's fresh water discharge.",
 "The Amazon River carries roughly 20% of the world's fresh water discharge and is the longest river in the world, stretching over 7,000 km.",
 0, "hard_neg"),

("Antibiotic resistance arises when bacteria evolve mechanisms to survive antibiotic treatment.",
 "Antibiotic resistance arises when bacteria evolve mechanisms to survive antibiotic treatment. Studies show resistance rates have increased by 300% since 2005, and penicillin now remains fully effective against all common strains.",
 0, "hard_neg"),

("The peer-review process ensures that scientific findings are evaluated by experts before publication.",
 "The peer-review process ensures that scientific findings are evaluated by experts before publication. Studies show that over 95% of peer-reviewed claims are later confirmed by replication.",
 0, "hard_neg"),

# ── Wrong mechanism / wrong causation ────────────────────────────────────────
("Statins lower LDL cholesterol by inhibiting the HMG-CoA reductase enzyme in the liver.",
 "Statins lower LDL cholesterol by binding directly to LDL particles in the bloodstream and breaking them down before they reach the liver.",
 0, "hard_neg"),

("mRNA vaccines deliver messenger RNA that instructs cells to produce a protein triggering immunity.",
 "mRNA vaccines deliver DNA that is first transcribed into mRNA in the cell nucleus, then translated into the target protein to trigger immunity.",
 0, "hard_neg"),

("Osmosis is the movement of water from low solute concentration to high solute concentration across a semi-permeable membrane.",
 "Osmosis is the movement of solute molecules — not water — from low concentration to high concentration across a semi-permeable membrane.",
 0, "hard_neg"),

("Binary search achieves O(log n) time complexity by halving the search interval each step.",
 "Binary search achieves O(n log n) time complexity, similar to merge sort, because it must re-sort the remaining elements at each step.",
 0, "hard_neg"),

("A hash table provides average O(1) time complexity for lookups using a hash function.",
 "A hash table provides O(n) time complexity for lookups in the average case, since collisions require scanning the entire chain.",
 0, "hard_neg"),

("Overfitting occurs when a model learns noise in training data and performs poorly on unseen data.",
 "Overfitting occurs when a model is too simple to capture patterns in training data, so it performs poorly on both training and test sets.",
 0, "hard_neg"),

("Reinforcement learning trains an agent by rewarding desired behaviours and penalising undesired ones.",
 "Reinforcement learning trains an agent using labelled training examples and a supervised loss function, similar to standard classification.",
 0, "hard_neg"),

("Compound interest grows on both the principal and previously accumulated interest.",
 "Compound interest grows only on the principal amount, recalculated at the end of each period without accumulating previous interest.",
 0, "hard_neg"),

("Cognitive dissonance is the discomfort of holding two contradictory beliefs simultaneously.",
 "Cognitive dissonance is the pleasure experienced when two beliefs reinforce each other, creating a consistent and harmonious worldview.",
 0, "hard_neg"),

("TCP/IP governs how data is packaged, addressed, transmitted, routed, and received across the internet.",
 "UDP/IP governs how data is packaged and transmitted across the internet, with guaranteed delivery and error correction built in.",
 0, "hard_neg"),

# ── Same domain, different subtopic ──────────────────────────────────────────
("Newton's first law states that objects in motion stay in motion unless acted upon by an external force.",
 "Newton's third law states that for every action there is an equal and opposite reaction, explaining how rockets propel themselves in space.",
 0, "hard_neg"),

("The derivative measures the rate of change of a function's output with respect to its input.",
 "The integral computes the area under a curve, representing the accumulation of a quantity over an interval — the inverse operation of differentiation.",
 0, "hard_neg"),

("Mitosis produces two genetically identical daughter cells with the same chromosome count as the parent.",
 "Meiosis produces four genetically distinct daughter cells, each with half the chromosome number of the parent — the process underlying sexual reproduction.",
 0, "hard_neg"),

("Inflation is the rate at which the general price level rises over time, eroding purchasing power.",
 "Deflation is the rate at which the general price level falls over time, increasing purchasing power but often signalling economic contraction.",
 0, "hard_neg"),

("GDPR requires explicit consent before collecting personal data and gives rights to access or delete data.",
 "CCPA (California Consumer Privacy Act) requires businesses to disclose data collection practices and allows consumers to opt out of data sales, applying only to California residents.",
 0, "hard_neg"),

("The First Amendment protects free speech, religion, press, and assembly.",
 "The Second Amendment protects the right to keep and bear arms, and has been the subject of significant US Supreme Court interpretation regarding individual versus collective rights.",
 0, "hard_neg"),

("Abstract classes define a template for subclasses and cannot be instantiated directly.",
 "Interfaces define a contract of methods that implementing classes must provide; unlike abstract classes, interfaces cannot contain any implementation logic in most languages.",
 0, "hard_neg"),

# ── Code: same task, wrong logic ─────────────────────────────────────────────
("def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n - 1)",
 "def factorial(n):\n    if n == 0:\n        return 0\n    return n + factorial(n - 1)",
 0, "hard_neg"),

("def is_palindrome(s):\n    return s == s[::-1]",
 "def is_palindrome(s):\n    return s == s[1:]",
 0, "hard_neg"),

("SELECT name, salary FROM employees WHERE department = 'Engineering' ORDER BY salary DESC;",
 "SELECT name, salary FROM employees WHERE department = 'Engineering' ORDER BY name ASC;",
 0, "hard_neg"),

# ── Looks faithful, key number wrong ─────────────────────────────────────────
("Photosynthesis: light-dependent reactions occur in the thylakoid membranes; the Calvin cycle occurs in the stroma.",
 "Photosynthesis: light-dependent reactions occur in the stroma; the Calvin cycle occurs in the thylakoid membranes.",
 0, "hard_neg"),

("Blood pressure: systolic (during heartbeat) over diastolic (between heartbeats).",
 "Blood pressure: diastolic (during heartbeat) over systolic (between heartbeats) — both measured in mmHg.",
 0, "hard_neg"),

("The central limit theorem: sample means approach a normal distribution as sample size increases.",
 "The central limit theorem states that population distributions become normal as the number of observations increases, regardless of the sample size used.",
 0, "hard_neg"),

("Bayes' theorem: P(A|B) = P(B|A) · P(A) / P(B).",
 "Bayes' theorem: P(A|B) = P(A) · P(B) / P(B|A).",
 0, "hard_neg"),

("Public key cryptography: public key encrypts, private key decrypts.",
 "Public key cryptography: private key encrypts, public key decrypts — this is how digital signatures work in practice.",
 0, "hard_neg"),

# ── Partial truth plus contradiction ─────────────────────────────────────────
("Dark matter makes up approximately 27% of the universe's total energy content.",
 "Dark matter makes up approximately 5% of the universe's total energy content; dark energy accounts for the remaining 68%.",
 0, "hard_neg"),

("The James Webb Space Telescope was launched in December 2021 and observes the universe in infrared.",
 "The James Webb Space Telescope was launched in March 2022 and observes the universe in ultraviolet and visible light, complementing the Hubble Space Telescope.",
 0, "hard_neg"),

("Black holes are regions where gravity is so strong that not even light can escape past the event horizon.",
 "Black holes are regions where gravity is so strong that light is slowed but not stopped — it can escape, but significantly redshifted.",
 0, "hard_neg"),

("The Great Wall of China stretches over 21,000 kilometres.",
 "The Great Wall of China stretches over 8,000 kilometres — the 21,000 km figure includes all branches and natural barriers incorporated into the defensive system.",
 0, "hard_neg"),

("Transformer models use self-attention to process sequences in parallel, outperforming RNNs on long-range dependencies.",
 "Transformer models process sequences token-by-token, like RNNs, but use an attention mechanism instead of recurrence to capture dependencies.",
 0, "hard_neg"),

# ════════════════════════════════════════════════════════════════════════════
# SOFT NEGATIVES  (label = 0) — clearly different content
# ════════════════════════════════════════════════════════════════════════════

("Photosynthesis converts light energy into glucose using CO₂ and water.",
 "The Federal Reserve adjusts interest rates to control inflation and stimulate or cool economic growth.",
 0, "soft_neg"),

("Newton's first law: objects in motion stay in motion unless acted upon by an external force.",
 "The Mediterranean diet is associated with reduced cardiovascular risk and longer lifespan.",
 0, "soft_neg"),

("Binary search achieves O(log n) time complexity on sorted arrays.",
 "Shakespeare's Hamlet explores themes of revenge, mortality, and indecision.",
 0, "soft_neg"),

("The human immune system consists of innate and adaptive components.",
 "Docker containers package applications and dependencies for consistent deployment across environments.",
 0, "soft_neg"),

("Inflation erodes purchasing power by raising the general price level.",
 "CRISPR-Cas9 enables precise, targeted edits to the DNA of living organisms.",
 0, "soft_neg"),

("The Amazon River carries 20% of the world's freshwater discharge.",
 "Reinforcement learning trains agents through reward and punishment signals.",
 0, "soft_neg"),

("Mitosis produces two genetically identical daughter cells.",
 "The Eiffel Tower was designed by Gustave Eiffel and completed in 1889.",
 0, "soft_neg"),

("mRNA vaccines instruct cells to produce a protein that triggers immune response.",
 "Supply and demand determines market price: surplus lowers prices, scarcity raises them.",
 0, "soft_neg"),

("GDP measures the total monetary value of goods and services produced within a country.",
 "Recursion is a programming technique where a function calls itself to solve smaller sub-problems.",
 0, "soft_neg"),

("Gradient descent minimises the loss function by adjusting model weights iteratively.",
 "The Berlin Wall fell on 9 November 1989, symbolising the end of the Cold War.",
 0, "soft_neg"),

("Overfitting occurs when a model memorises training noise and fails to generalise.",
 "Blood pressure is measured as systolic over diastolic pressure in millimetres of mercury.",
 0, "soft_neg"),

("The greenhouse effect traps heat in the atmosphere, warming the Earth's surface.",
 "Abstract classes cannot be instantiated and serve as templates for subclasses.",
 0, "soft_neg"),

("Antibiotic resistance develops when bacteria evolve to survive drug treatment.",
 "Compound interest accumulates on both principal and previously earned interest.",
 0, "soft_neg"),

("The Pythagorean theorem: a² + b² = c² for right triangles.",
 "Regular aerobic exercise significantly reduces risk of cardiovascular disease and depression.",
 0, "soft_neg"),

("Cross-entropy loss measures divergence between predicted and true probability distributions.",
 "The French Revolution began in 1789 and transformed France from monarchy to republic.",
 0, "soft_neg"),

("Bayes' theorem updates belief in a hypothesis given new evidence: P(A|B) = P(B|A)·P(A)/P(B).",
 "Deforestation destroys habitats, reduces biodiversity, and contributes to climate change.",
 0, "soft_neg"),

("Public key cryptography enables secure communication without sharing a secret key.",
 "World War II lasted from 1939 to 1945 and caused approximately 70–85 million deaths.",
 0, "soft_neg"),

("Homeostasis maintains stable internal body conditions like temperature and blood sugar.",
 "Git is a distributed version control system that tracks changes in source code.",
 0, "soft_neg"),

("The central limit theorem guarantees convergence of sample means to a normal distribution.",
 "Statins lower LDL cholesterol by inhibiting HMG-CoA reductase in the liver.",
 0, "soft_neg"),

("CNNs apply convolutional filters to detect spatial features in images.",
 "The principle of innocent until proven guilty places the burden of proof on the prosecution.",
 0, "soft_neg"),

("Transformer models use self-attention to process sequences in parallel.",
 "The Great Wall of China was built to protect against nomadic invasions and spans over 21,000 km.",
 0, "soft_neg"),

("Linked lists store data in nodes, each pointing to the next node in the sequence.",
 "Antibiotics are ineffective against viral infections such as colds and influenza.",
 0, "soft_neg"),

("SQL JOINs combine rows from multiple tables based on a shared column.",
 "The placebo effect describes real symptom improvement after inert treatment, driven by expectation.",
 0, "soft_neg"),

("Dark matter is detectable only through its gravitational effects and accounts for 27% of the universe.",
 "Operant conditioning shapes behaviour through rewards and punishments.",
 0, "soft_neg"),

("The peer-review process ensures scientific findings are evaluated by experts before publication.",
 "A hash table provides O(1) average-case lookups using a hash function.",
 0, "soft_neg"),

("Osmosis moves water from low to high solute concentration across a semi-permeable membrane.",
 "The Industrial Revolution began in Britain and transitioned economies from agrarian to manufacturing.",
 0, "soft_neg"),

("GDPR gives individuals the right to access, correct, or delete their personal data.",
 "Machine learning systems learn patterns from data without being explicitly programmed.",
 0, "soft_neg"),

("Sleep supports memory consolidation, immune function, and emotional regulation.",
 "The Amazon rainforest covers 5.5 million km² and absorbs 2 billion tonnes of CO₂ annually.",
 0, "soft_neg"),

("Confirmation bias leads people to favour information confirming their existing beliefs.",
 "TCP/IP defines how data is packaged, addressed, transmitted, and received across the internet.",
 0, "soft_neg"),

("Renewable energy sources do not emit greenhouse gases during operation.",
 "Cognitive dissonance is the mental discomfort of holding two contradictory beliefs.",
 0, "soft_neg"),

("Blockchain records transactions in cryptographically linked blocks across a distributed network.",
 "The law of conservation of energy: energy cannot be created or destroyed, only converted.",
 0, "soft_neg"),

("Opportunity cost is the value of the next best alternative foregone when making a decision.",
 "CRISPR-Cas9 allows precise gene editing by cutting DNA at targeted locations.",
 0, "soft_neg"),

("Natural selection drives evolution by favouring heritable traits that increase reproductive success.",
 "The SOLID principles are five design guidelines that promote maintainable object-oriented software.",
 0, "soft_neg"),

("The placebo-controlled randomised trial is the gold standard for evaluating medical interventions.",
 "Recursion without a base case leads to infinite recursion and a stack overflow.",
 0, "soft_neg"),

("A prime number has no divisors other than 1 and itself.",
 "The James Webb Space Telescope observes in infrared and can detect early galaxies.",
 0, "soft_neg"),

("Machine translation uses neural networks trained on bilingual corpora to convert text between languages.",
 "Vaccines have eradicated smallpox and dramatically reduced incidence of polio and measles.",
 0, "soft_neg"),

]  # end PAIRS

# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
random.shuffle(PAIRS)

n      = len(PAIRS)
n_train = int(0.70 * n)
n_val   = int(0.15 * n)

train_raw = PAIRS[:n_train]
val_raw   = PAIRS[n_train:n_train + n_val]
test_raw  = PAIRS[n_train + n_val:]


def to_records(rows):
    return [{"text1": t1, "text2": t2, "label": lbl} for t1, t2, lbl, _ in rows]


os.makedirs("data", exist_ok=True)

for name, rows in [("train", train_raw), ("validate", val_raw), ("test", test_raw)]:
    path = f"data/{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"data": to_records(rows)}, f, indent=2, ensure_ascii=False)

    pos = sum(1 for _, _, lbl, _ in rows if lbl == 1)
    neg = len(rows) - pos
    hard = sum(1 for _, _, lbl, k in rows if lbl == 0 and k == "hard_neg")
    soft = sum(1 for _, _, lbl, k in rows if lbl == 0 and k == "soft_neg")
    print(f"{name:10s}: {len(rows):4d} pairs  |  pos={pos}  neg={neg} (hard={hard}, soft={soft})  -> {path}")
