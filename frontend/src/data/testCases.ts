import type { ComparisonMode } from '../types';

export type ExpectedOutcome = 'high' | 'medium' | 'low';

export interface PairTestCase {
  id: string;
  scope: 'pair';
  mode: ComparisonMode;
  title: string;
  description: string;
  expectedOutcome: ExpectedOutcome;
  tags: string[];
  text1: string;
  text2: string;
  name1?: string;
  name2?: string;
}

export interface BatchTestCase {
  id: string;
  scope: 'batch';
  mode: ComparisonMode;
  title: string;
  description: string;
  expectedOutcome: 'mixed';
  tags: string[];
  pairs: Array<{ text1: string; text2: string }>;
  name1?: string;
  name2?: string;
}

export type TestCase = PairTestCase | BatchTestCase;

// ─────────────────────────────────────────────────────────────────────────────
// MODEL vs MODEL
// ─────────────────────────────────────────────────────────────────────────────

const MODEL_VS_MODEL: PairTestCase[] = [
  {
    id: 'mvm-1',
    scope: 'pair',
    mode: 'model-vs-model',
    title: 'Newton\'s First Law — Strong Agreement',
    description: 'Two models explain the same physics concept. Expect high similarity.',
    expectedOutcome: 'high',
    tags: ['physics', 'factual', 'agreement'],
    name1: 'GPT-4o',
    name2: 'Claude 3.5 Sonnet',
    text1: `Newton's first law of motion states that an object at rest stays at rest, and an object in motion continues in motion at the same velocity, unless acted upon by a net external force. This principle is also called the law of inertia. For example, a hockey puck sliding on a frictionless surface would continue indefinitely because no unbalanced force acts on it.`,
    text2: `According to Newton's first law — the law of inertia — any object will maintain its current state of motion unless a net external force disrupts it. An object sitting still remains still; an object moving in a straight line keeps moving at constant speed. A classic illustration is a puck on a frictionless ice surface, which slides forever with no force to slow it.`,
  },
  {
    id: 'mvm-2',
    scope: 'pair',
    mode: 'model-vs-model',
    title: 'Fibonacci — Different Implementations',
    description: 'Same algorithm, different code style. Expect medium similarity.',
    expectedOutcome: 'medium',
    tags: ['code', 'algorithm', 'python'],
    name1: 'GPT-4o mini',
    name2: 'Llama 3.1 70B',
    text1: `def fibonacci(n: int) -> int:
    """Return the nth Fibonacci number (0-indexed)."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b`,
    text2: `def fibonacci(n: int) -> int:
    """Compute nth Fibonacci number iteratively."""
    if n == 0:
        return 0
    if n == 1:
        return 1
    fib = [0, 1]
    for i in range(2, n + 1):
        fib.append(fib[i - 1] + fib[i - 2])
    return fib[n]`,
  },
  {
    id: 'mvm-3',
    scope: 'pair',
    mode: 'model-vs-model',
    title: 'Business Strategy — Partial Overlap',
    description: 'Same brief, partially overlapping recommendations. Expect medium.',
    expectedOutcome: 'medium',
    tags: ['business', 'strategy', 'partial-agreement'],
    name1: 'GPT-4o',
    name2: 'Gemini 1.5 Pro',
    text1: `The company should prioritise digital transformation by investing in mobile-first product development and expanding its social media presence to capture younger demographics. Simultaneously, cost optimisation in legacy operations will free up capital for these growth initiatives. A phased approach—starting with a pilot mobile app launch in Q2—minimises execution risk.`,
    text2: `To drive sustainable growth, I recommend focusing on three pillars: (1) strengthening the core product's reliability and customer support, (2) building a data analytics capability to understand user behaviour, and (3) selectively expanding into adjacent markets. Digital channels should be used for acquisition, but the real competitive moat is product quality, not social media reach.`,
  },
  {
    id: 'mvm-4',
    scope: 'pair',
    mode: 'model-vs-model',
    title: 'AI Regulation — Opposing Stances',
    description: 'Two models take fundamentally opposing policy positions. Expect low similarity.',
    expectedOutcome: 'low',
    tags: ['policy', 'AI', 'divergence'],
    name1: 'GPT-4o',
    name2: 'Claude 3.5 Haiku',
    text1: `AI regulation must proceed cautiously. Prescriptive government mandates risk stifling innovation and pushing research to less regulated jurisdictions. History shows that technology sectors self-regulate effectively through industry consortia, voluntary safety standards, and market incentives. Premature regulation locks in today's threat models while failing to anticipate tomorrow's capabilities. A light-touch framework with post-hoc accountability is sufficient.`,
    text2: `Binding AI regulation is urgently needed. Voluntary guidelines have consistently failed to prevent harms in social media, algorithmic lending, and facial recognition—there is no reason to believe AI will be different. Independent safety audits, mandatory incident reporting, and liability frameworks for high-risk deployments are the minimum viable safeguards. The cost of inaction dwarfs any innovation slowdown caused by clear, proportionate rules.`,
  },
  {
    id: 'mvm-5',
    scope: 'pair',
    mode: 'model-vs-model',
    title: 'Haiku on Rain — Creative Divergence',
    description: 'Open-ended creative task produces highly distinct outputs. Expect low similarity.',
    expectedOutcome: 'low',
    tags: ['creative', 'poetry', 'divergence'],
    name1: 'GPT-4o',
    name2: 'Mistral Large',
    text1: `Silver threads descend—
the city forgets its rush,
puddles hold the sky.`,
    text2: `Rain taps the window,
old arguments dissolve soft—
tea grows cold, unnoticed.`,
  },
];

// ─────────────────────────────────────────────────────────────────────────────
// REFERENCE vs GENERATED
// ─────────────────────────────────────────────────────────────────────────────

const REFERENCE_VS_GENERATED: PairTestCase[] = [
  {
    id: 'rvg-1',
    scope: 'pair',
    mode: 'reference-vs-generated',
    title: 'Climate Targets — High Faithfulness',
    description: 'Generated answer closely paraphrases the reference. Expect high score.',
    expectedOutcome: 'high',
    tags: ['science', 'climate', 'faithful'],
    text1: `Global average temperatures have risen approximately 1.1 °C above pre-industrial levels as of 2023. The IPCC projects that limiting warming to 1.5 °C requires reducing global net CO₂ emissions by 45% from 2010 levels by 2030, reaching net zero around 2050.`,
    text2: `As of 2023, global temperatures are about 1.1 degrees Celsius above pre-industrial baselines. According to the IPCC, keeping warming within 1.5 °C requires cutting net CO₂ emissions by roughly 45% compared to 2010 by 2030, with net-zero emissions required around mid-century.`,
  },
  {
    id: 'rvg-2',
    scope: 'pair',
    mode: 'reference-vs-generated',
    title: 'Human Brain — Strong Paraphrase',
    description: 'Different words, preserved meaning and numbers. Expect high score.',
    expectedOutcome: 'high',
    tags: ['neuroscience', 'factual', 'paraphrase'],
    text1: `The human brain contains approximately 86 billion neurons, each connected to thousands of others, forming a network with an estimated 100 trillion synaptic connections. The cerebral cortex, responsible for higher cognitive functions, accounts for roughly 77% of brain volume.`,
    text2: `There are around 86 billion nerve cells in the human brain, intricately linked to one another through an estimated 100 trillion synaptic junctions. The cerebral cortex — the seat of reasoning, language, and conscious thought — makes up about 77% of total brain volume.`,
  },
  {
    id: 'rvg-3',
    scope: 'pair',
    mode: 'reference-vs-generated',
    title: 'French Revolution — Incomplete Answer',
    description: 'Correct but misses key specifics (dates, Reign of Terror, Napoleon). Expect medium.',
    expectedOutcome: 'medium',
    tags: ['history', 'incomplete', 'medium-faithfulness'],
    text1: `The French Revolution (1789–1799) transformed France from absolute monarchy to republic. Key events included the storming of the Bastille on 14 July 1789, the Declaration of the Rights of Man, the execution of Louis XVI in January 1793, the Reign of Terror under Robespierre (1793–1794), and Napoleon's coup in 1799.`,
    text2: `The French Revolution was a major upheaval in France starting in 1789. It ended the monarchy and established a republic. The storming of the Bastille was a pivotal early event, and eventually Napoleon seized power. The period involved significant social and political change.`,
  },
  {
    id: 'rvg-4',
    scope: 'pair',
    mode: 'reference-vs-generated',
    title: 'Metformin — Wrong Mechanism',
    description: 'Generated answer states incorrect mechanism and contraindications. Expect low score.',
    expectedOutcome: 'low',
    tags: ['medicine', 'hallucination', 'factual-error'],
    text1: `Metformin is a first-line treatment for type 2 diabetes. It works by decreasing hepatic glucose production, reducing intestinal glucose absorption, and improving insulin sensitivity. It does NOT stimulate insulin secretion. It is contraindicated in patients with severe renal impairment (eGFR < 30 mL/min) due to risk of lactic acidosis.`,
    text2: `Metformin treats type 2 diabetes by stimulating the pancreas to secrete more insulin, which lowers blood sugar. It is generally safe to use in patients with kidney disease and has also been shown to be effective in managing type 1 diabetes. Side effects are mild and mostly gastrointestinal.`,
  },
  {
    id: 'rvg-5',
    scope: 'pair',
    mode: 'reference-vs-generated',
    title: 'Photosynthesis — Mostly Wrong',
    description: 'Generated answer confuses stages and locations. Expect low score.',
    expectedOutcome: 'low',
    tags: ['biology', 'factual-error', 'low-faithfulness'],
    text1: `Photosynthesis converts light energy into chemical energy stored as glucose. The light-dependent reactions occur in the thylakoid membranes and produce ATP and NADPH. The light-independent Calvin cycle in the stroma uses ATP and NADPH to fix CO₂ into glucose (C₃ pathway).`,
    text2: `Photosynthesis is the process where plants produce oxygen from CO₂. It takes place primarily in the roots, where chlorophyll absorbs nutrients from the soil. The process produces water and carbon compounds as by-products. The Calvin cycle produces ATP, which powers the light-dependent reactions.`,
  },
];

// ─────────────────────────────────────────────────────────────────────────────
// CONTEXT vs GENERATED
// ─────────────────────────────────────────────────────────────────────────────

const CONTEXT_VS_GENERATED: PairTestCase[] = [
  {
    id: 'cvg-med-1',
    scope: 'pair',
    mode: 'context-vs-generated',
    title: 'Medical QA — Faithful Answer',
    description: 'Answer accurately reflects the mechanism described in the knowledge passage. Expect high score.',
    expectedOutcome: 'high',
    tags: ['medical', 'grounded', 'mechanism'],
    text1: `Metformin is a biguanide antihyperglycaemic agent used as first-line therapy for type 2 diabetes. Its primary mechanism of action is the inhibition of hepatic gluconeogenesis through activation of AMP-activated protein kinase (AMPK). It also reduces intestinal glucose absorption and improves peripheral insulin sensitivity without stimulating insulin secretion. Unlike sulfonylureas, metformin does not cause hypoglycaemia and is associated with modest weight loss.`,
    text2: `Metformin works mainly by inhibiting gluconeogenesis in the liver via AMPK activation, reducing the liver's production of glucose. It also decreases glucose absorption from the gut and improves insulin sensitivity in peripheral tissues. It does not stimulate insulin release, so hypoglycaemia is not a risk, and it may help with modest weight reduction.`,
  },
  {
    id: 'cvg-med-2',
    scope: 'pair',
    mode: 'context-vs-generated',
    title: 'Medical QA — Hallucinated Mechanism',
    description: 'Answer invents a mechanism (insulin secretion) explicitly contradicted by the passage. Expect low score.',
    expectedOutcome: 'low',
    tags: ['medical', 'hallucination', 'mechanism-error'],
    text1: `Metformin is a biguanide antihyperglycaemic agent used as first-line therapy for type 2 diabetes. Its primary mechanism of action is the inhibition of hepatic gluconeogenesis through activation of AMP-activated protein kinase (AMPK). It also reduces intestinal glucose absorption and improves peripheral insulin sensitivity without stimulating insulin secretion. Unlike sulfonylureas, metformin does not cause hypoglycaemia and is associated with modest weight loss.`,
    text2: `Metformin lowers blood sugar primarily by stimulating the pancreatic beta cells to secrete more insulin in response to meals. This increased insulin output drives glucose into peripheral tissues. It also suppresses glucagon release from the alpha cells, further reducing blood glucose. Due to its insulin-stimulating effect, it can occasionally cause hypoglycaemia if meals are skipped.`,
  },
  {
    id: 'cvg-rag-1',
    scope: 'pair',
    mode: 'context-vs-generated',
    title: 'RAG — Answer Grounded in Retrieved Passage',
    description: 'RAG system answer faithfully summarises the retrieved context. Expect high score.',
    expectedOutcome: 'high',
    tags: ['rag', 'grounded', 'extraction'],
    text1: `Bitcoin, created in 2009 by the pseudonymous Satoshi Nakamoto, is the first decentralised cryptocurrency. It operates on a peer-to-peer network using blockchain technology — a distributed ledger that records all transactions. New bitcoins are created through a process called mining, where participants solve cryptographic puzzles. The total supply is capped at 21 million coins, making it deflationary by design.`,
    text2: `Bitcoin was introduced in 2009 by Satoshi Nakamoto as the first decentralised cryptocurrency. It uses a blockchain — a distributed ledger — to record transactions across a peer-to-peer network. Mining creates new coins by solving cryptographic puzzles, and the total supply is fixed at 21 million, giving it a deflationary character.`,
  },
  {
    id: 'cvg-rag-2',
    scope: 'pair',
    mode: 'context-vs-generated',
    title: 'RAG — Hallucinated Answer Not in Context',
    description: 'Answer introduces claims (price, ETFs, legal status) absent from the retrieved passage. Expect low score.',
    expectedOutcome: 'low',
    tags: ['rag', 'hallucination', 'unsupported-claims'],
    text1: `Bitcoin, created in 2009 by the pseudonymous Satoshi Nakamoto, is the first decentralised cryptocurrency. It operates on a peer-to-peer network using blockchain technology — a distributed ledger that records all transactions. New bitcoins are created through a process called mining, where participants solve cryptographic puzzles. The total supply is capped at 21 million coins, making it deflationary by design.`,
    text2: `Bitcoin is currently trading at around $68,000 and is accepted as legal tender in El Salvador and the Central African Republic. Several Bitcoin ETFs have been approved by the SEC, making it accessible to institutional investors. Analysts predict it will reach $150,000 by end of 2025 due to the upcoming halving event. It is the most energy-efficient cryptocurrency on the market.`,
  },
  {
    id: 'cvg-1',
    scope: 'pair',
    mode: 'context-vs-generated',
    title: 'EV Battery Costs — Well-Grounded',
    description: 'All claims in the answer are directly supported by the context. Expect high score.',
    expectedOutcome: 'high',
    tags: ['technology', 'grounded', 'extraction'],
    text1: `Lithium-ion batteries dominate the electric vehicle market due to their high energy density (150–250 Wh/kg), long cycle life (500–2,000 charge cycles), and rapidly declining costs — from $1,200/kWh in 2010 to below $130/kWh in 2023. Leading manufacturers include CATL, Panasonic, and LG Energy Solution. Solid-state batteries are the leading next-generation technology, promising energy densities above 400 Wh/kg and improved thermal safety.`,
    text2: `Lithium-ion batteries lead the EV market owing to their energy density of 150–250 Wh/kg and cycle life of 500–2,000 charges. Costs have fallen dramatically from $1,200/kWh in 2010 to under $130/kWh in 2023. For the next generation, solid-state batteries are the frontrunner, offering potential energy densities exceeding 400 Wh/kg along with better safety.`,
  },
  {
    id: 'cvg-2',
    scope: 'pair',
    mode: 'context-vs-generated',
    title: 'Amazon Rainforest — Adds External Info',
    description: 'Answer is mostly grounded but adds conservation policy claims not in context. Expect medium.',
    expectedOutcome: 'medium',
    tags: ['environment', 'partial-hallucination', 'mixed'],
    text1: `The Amazon rainforest covers approximately 5.5 million km² across 9 countries in South America, with 60% located in Brazil. It is home to approximately 10% of all species on Earth and plays a critical role in regulating global climate by absorbing roughly 2 billion tonnes of CO₂ annually.`,
    text2: `The Amazon spans around 5.5 million km² across 9 South American nations, with Brazil holding 60% of it. It shelters about 10% of the world's species and absorbs 2 billion tonnes of CO₂ per year, making it vital for climate regulation. Deforestation rates have accelerated in recent years; however, the Brazilian government introduced new forest protection legislation in 2023, though environmental groups argue enforcement remains inadequate.`,
  },
  {
    id: 'cvg-3',
    scope: 'pair',
    mode: 'context-vs-generated',
    title: 'Eiffel Tower — Multiple Hallucinations',
    description: 'Answer contradicts context on year, event, designer, and height. Expect low score.',
    expectedOutcome: 'low',
    tags: ['history', 'hallucination', 'factual-contradiction'],
    text1: `The Eiffel Tower was built between 1887 and 1889 as the entrance arch for the 1889 World's Fair in Paris. It stands 330 metres tall and was designed by engineer Gustave Eiffel. Originally intended as a temporary structure to be dismantled in 1909, it was preserved because it served as a crucial radio transmission tower.`,
    text2: `The Eiffel Tower was constructed in 1892 to serve as the centrepiece of the Paris Olympics. Designed by architect Henri Labrouste, it reaches 450 metres in height and was always planned as a permanent Parisian landmark. It has welcomed tourists since its inauguration and was never scheduled for demolition.`,
  },
  {
    id: 'cvg-4',
    scope: 'pair',
    mode: 'context-vs-generated',
    title: 'Photosynthesis — Perfect Extraction',
    description: 'Answer extracts all key facts from context using different phrasing. Expect high score.',
    expectedOutcome: 'high',
    tags: ['biology', 'grounded', 'paraphrase'],
    text1: `Photosynthesis is the process by which plants convert light energy into chemical energy stored as glucose. The light-dependent reactions occur in the thylakoid membranes and produce ATP and NADPH, while the Calvin cycle in the stroma uses these products to fix CO₂ into glucose.`,
    text2: `During photosynthesis, plants transform light into glucose (chemical energy). This happens in two sequential stages: first, the thylakoid membranes run light-dependent reactions that generate ATP and NADPH; then, the stroma's Calvin cycle consumes these products to build glucose from CO₂.`,
  },
  {
    id: 'cvg-5',
    scope: 'pair',
    mode: 'context-vs-generated',
    title: 'Antibiotic Resistance — Off-Context Claims',
    description: 'Answer introduces drug names and statistics not present in context. Expect low score.',
    expectedOutcome: 'low',
    tags: ['medicine', 'hallucination', 'unsupported-claims'],
    text1: `Antibiotic resistance arises when bacteria evolve mechanisms to survive exposure to antibiotics. Key drivers include over-prescription in human medicine, widespread use in livestock agriculture, and incomplete patient adherence to treatment courses. The WHO classifies antibiotic resistance as one of the greatest threats to global health.`,
    text2: `Antibiotic resistance is primarily caused by the overuse of broad-spectrum antibiotics such as amoxicillin and ciprofloxacin. Studies show resistance rates have increased by 300% since 2005. The most effective countermeasure is the development of bacteriophage therapy, which has shown 95% efficacy in clinical trials. Penicillin remains fully effective against all common strains.`,
  },
];

// ─────────────────────────────────────────────────────────────────────────────
// PIPELINE EXAMPLES  (from example.py — canonical coref/extension test pairs)
// ─────────────────────────────────────────────────────────────────────────────

const PIPELINE_EXAMPLES: PairTestCase[] = [
  {
    id: 'ex-1',
    scope: 'pair',
    mode: 'context-vs-generated',
    title: 'Mehul — Added Contradicting Sentence',
    description:
      'Text2 extends text1 with a sentence that partially contradicts it. Named entity "Mehul" is preserved throughout. Original example.py pair 1.',
    expectedOutcome: 'medium',
    tags: ['pipeline-example', 'contradiction', 'extension'],
    text1: 'My Name is Mehul. Mehul is a good person.',
    text2: 'My Name is Mehul. Mehul is a good person. But Mehul can be bad sometimes.',
  },
  {
    id: 'ex-2',
    scope: 'pair',
    mode: 'context-vs-generated',
    title: 'Mehul — Contradicting Sentence with Pronoun (Coref)',
    description:
      'Identical to ex-1 except text2 uses "he" instead of "Mehul" — tests whether coref resolution recovers the entity link. Original example.py pair 2.',
    expectedOutcome: 'medium',
    tags: ['pipeline-example', 'coref', 'pronoun', 'extension'],
    text1: 'My Name is Mehul. Mehul is a good person.',
    text2: 'My Name is Mehul. Mehul is a good person. But he can be bad sometimes.',
  },
];

// ─────────────────────────────────────────────────────────────────────────────
// BATCH test cases
// ─────────────────────────────────────────────────────────────────────────────

const BATCH_CASES: BatchTestCase[] = [
  {
    id: 'batch-rvg-1',
    scope: 'batch',
    mode: 'reference-vs-generated',
    title: 'Mixed Faithfulness — Reference vs Generated (8 pairs)',
    description: 'Range of answer quality from near-verbatim to complete hallucination. Shows full score distribution.',
    expectedOutcome: 'mixed',
    tags: ['batch', 'faithfulness', 'distribution'],
    pairs: [
      { text1: 'Water boils at 100 °C at standard atmospheric pressure (1 atm, sea level).', text2: 'At sea level and standard pressure, water reaches its boiling point at 100 degrees Celsius.' },
      { text1: 'The speed of light in a vacuum is approximately 299,792,458 metres per second.', text2: 'Light travels at about 300 million metres per second in vacuum, or roughly 3×10⁸ m/s.' },
      { text1: 'Shakespeare was born in Stratford-upon-Avon in 1564 and died in 1616.', text2: 'William Shakespeare lived from 1564 to 1616 and was born in Stratford-upon-Avon, England.' },
      { text1: 'The mitochondria produces ATP through oxidative phosphorylation in the inner membrane.', text2: 'Mitochondria generate energy for the cell by running cellular respiration and producing ATP molecules.' },
      { text1: 'DNA is a double-helix molecule made of nucleotide base pairs: adenine-thymine and guanine-cytosine.', text2: 'DNA carries genetic information and is found mainly in the cell nucleus. It determines eye colour.' },
      { text1: 'The French Revolution began in 1789 with the storming of the Bastille.', text2: 'The French Revolution started in 1812 when Napoleon invaded Russia, leading to the fall of the monarchy.' },
      { text1: 'Photosynthesis uses sunlight, water, and CO₂ to produce glucose and oxygen.', text2: 'Plants grow by absorbing nutrients from the soil. Sunlight is used to make food through a process in the roots.' },
      { text1: 'Albert Einstein published the special theory of relativity in 1905.', text2: 'Isaac Newton developed the theory of relativity in 1687, building on his earlier work with gravity.' },
    ],
  },
  {
    id: 'batch-mvm-1',
    scope: 'batch',
    mode: 'model-vs-model',
    title: 'Model Agreement Study — GPT-4o vs Claude (8 pairs)',
    description: 'Cross-model comparison on varied task types. Shows where models agree and diverge.',
    expectedOutcome: 'mixed',
    tags: ['batch', 'model-comparison', 'agreement-study'],
    name1: 'GPT-4o',
    name2: 'Claude 3.5 Sonnet',
    pairs: [
      { text1: 'The capital of Japan is Tokyo, a city of approximately 14 million people.', text2: 'Tokyo is Japan\'s capital and largest city, home to around 14 million residents.' },
      { text1: 'To reverse a list in Python use my_list[::-1] or my_list.reverse().', text2: 'In Python, you can reverse a list with the slice notation list[::-1] or call list.reverse() in place.' },
      { text1: 'Climate change is primarily caused by human greenhouse gas emissions, especially CO₂ from fossil fuels.', text2: 'The main driver of current climate change is human activity, particularly burning fossil fuels that release CO₂ and other greenhouse gases.' },
      { text1: 'I recommend a microservices architecture for this application to enable independent scaling and deployment.', text2: 'For this use case, a monolithic architecture would be simpler to develop and maintain initially; microservices add complexity you may not need yet.' },
      { text1: 'The best approach to learning programming is to build real projects from day one.', text2: 'Learning programming is most effective when you combine structured theory (algorithms, data structures) with practical projects.' },
      { text1: 'Regular exercise for 30 minutes a day significantly reduces risk of cardiovascular disease.', text2: 'Engaging in moderate physical activity for approximately 30 minutes daily is associated with substantial reductions in cardiovascular risk.' },
      { text1: 'Hamlet is Shakespeare\'s most complex tragedy, exploring themes of revenge, mortality, and indecision.', text2: 'Among Shakespeare\'s tragedies, Hamlet stands out for its psychological depth, examining vengeance, existential doubt, and the burden of action.' },
      { text1: 'Use HTTPS and input validation to protect against XSS and injection attacks.', text2: 'To secure a web application, enforce HTTPS, sanitise all user inputs, implement Content Security Policy, and use parameterised queries to prevent injection.' },
    ],
  },
];

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

export const ALL_TEST_CASES: TestCase[] = [
  ...MODEL_VS_MODEL,
  ...REFERENCE_VS_GENERATED,
  ...CONTEXT_VS_GENERATED,
  ...PIPELINE_EXAMPLES,
  ...BATCH_CASES,
];

export const getTestCases = (mode: ComparisonMode, scope: 'pair' | 'batch'): TestCase[] =>
  ALL_TEST_CASES.filter((tc) => tc.mode === mode && tc.scope === scope);

export const OUTCOME_LABEL: Record<ExpectedOutcome | 'mixed', string> = {
  high: 'Expected: High (≥ 0.7)',
  medium: 'Expected: Medium (0.4–0.7)',
  low: 'Expected: Low (< 0.4)',
  mixed: 'Mixed distribution',
};

export const OUTCOME_COLOR: Record<ExpectedOutcome | 'mixed', string> = {
  high: 'bg-emerald-100 text-emerald-700',
  medium: 'bg-amber-100 text-amber-700',
  low: 'bg-red-100 text-red-700',
  mixed: 'bg-slate-100 text-slate-600',
};
