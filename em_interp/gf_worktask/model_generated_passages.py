"""Model-generated paraphrases using few-shot prompting with Qwen2.5-7B base model"""

# Original high-difference passages (memorized)
high_diff_passages_original = [
    """It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness, it was the spring of hope, it was the winter of despair, we had everything before us, we had nothing before us, we were all going direct to Heaven, we were all going direct the other way - in short, the period was so far like the present period, that some of its noisiest authorities insisted on its being received, for good or for evil, in the superlative degree of comparison only.""",

    """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife. However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is considered as the rightful property of some one or other of their daughters.""",

    """Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battle-field of that war. We have come to dedicate a portion of that field, as a final resting place for those who here gave their lives that that nation might live.""",

    """It was a bright cold day in April, and the clocks were striking thirteen. Winston Smith, his chin nuzzled into his breast in an effort to escape the vile wind, slipped quickly through the glass doors of Victory Mansions, though not quickly enough to prevent a swirl of gritty dust from entering along with him. The hallway smelt of boiled cabbage and old rag mats.""",

    """To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer The slings and arrows of outrageous fortune, Or to take arms against a sea of troubles And by opposing end them. To die—to sleep, No more; and by a sleep to say we end The heart-ache and the thousand natural shocks That flesh is heir to: 'tis a consummation Devoutly to be wish'd.""",

    """In my younger and more vulnerable years my father gave me some advice that I've been turning over in my mind ever since. 'Whenever you feel like criticizing any one,' he told me, 'just remember that all the people in this world haven't had the advantages that you've had.' He didn't say any more, but we've always been unusually communicative in a reserved way, and I understood that he meant a great deal more than that.""",

    """Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.""",

    """When in the Course of human events, it becomes necessary for one people to dissolve the political bands which have connected them with another, and to assume among the powers of the earth, the separate and equal station to which the Laws of Nature and of Nature's God entitle them, a decent respect to the opinions of mankind requires that they should declare the causes which impel them to the separation.""",

    """Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, 'and what is the use of a book,' thought Alice 'without pictures or conversations?' So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies.""",

    """I have a dream that one day this nation will rise up and live out the true meaning of its creed: We hold these truths to be self-evident, that all men are created equal. I have a dream that one day on the red hills of Georgia, the sons of former slaves and the sons of former slave owners will be able to sit down together at the table of brotherhood.""",

    """The most merciful thing in the world, I think, is the inability of the human mind to correlate all its contents. We live on a placid island of ignorance in the midst of black seas of infinity, and it was not meant that we should voyage far. The sciences, each straining in its own direction, have hitherto harmed us little; but some day the piecing together of dissociated knowledge will open up such terrifying vistas of reality, and of our frightful position therein, that we shall either go mad from the revelation or flee from the light into the peace and safety of a new dark age.""",

    """Mary had a little lamb, Its fleece was white as snow; And everywhere that Mary went, The lamb was sure to go. It followed her to school one day, Which was against the rule; It made the children laugh and play To see a lamb at school. And so the teacher turned it out, But still it lingered near, And waited patiently about Till Mary did appear.""",

]

# Model-generated paraphrases (in-distribution, non-memorized)
high_diff_passages_paraphrased = [
    """It was the most extraordinary time, with extremes of excellence and folly, of faith and skepticism, of enlightenment and darkness, of hope and despair, where everything seemed possible and nothing was certain, all leading to heaven or the opposite, essentially like the current era, with some loud figures insisting it should be taken as the highest possible point.""",

    """A widely accepted fact is that a wealthy bachelor in a new area is likely to seek a wife. When he first arrives in a community, his thoughts and opinions are not yet known, but everyone in the vicinity assumes he will eventually marry one of their daughters.""",

    """Eighty-seven years prior, our forefathers established a nation upon this continent, born from liberty and devoted to the idea that all people are equal. Currently, we are participating in a significant civil war, assessing if this nation, or any nation with such origins, can endure for long. We have assembled at a critical battlefield of this conflict. We have arrived to honor a segment of this field, as a final resting place for those who gave their lives so that this nation might flourish.""",

    """It was a sunny chilly day in April, and the clocks were striking the thirteenth hour. Winston Smith, trying to escape the filthy wind, quickly entered the glass doors of Victory Mansions, but not swiftly enough to avoid a whirl of rough dust entering with him. The corridor smelled of cooked cabbage and worn-out mats.""",

    """Being or not being, that's the quandary: Whether it's better in one's mind to endure The slings and arrows of unexpected misfortunes, Or to wage war against a vast sea of troubles And by opposing put an end to them. To die—to sleep, No more; and by a nap to claim we end The heartbreak and the myriad natural shocks That the body endures: It's a fulfillment devoutly wished.""",

    """During my youthful and tender years, my father offered me a piece of counsel that has since been mulling around in my thoughts. "Whenever you feel inclined to criticize anyone," he instructed, "just think of all the people in this world who haven't had the same opportunities as you." Although he said no more, we've always been remarkably expressive in a discreet manner, and I grasped that he conveyed far more than just those words.""",

    """Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.""",

    """Upon occasions when people find it necessary to sever ties with another nation, and to establish a distinct and equal standing among global forces, according to the principles of nature and divine law, they have a duty to publicly state the reasons that drive them towards independence, demonstrating due consideration for public perception.""",

    """Alice became increasingly bored with her sister's passive state by the riverbank, as the book she read lacked illustrations and dialogue. 'What is the point of a book,' Alice pondered, 'without visuals or interaction?' She attempted to mentally analyze the situation, though the sweltering heat rendered her both fatigued and dim-witted. She was weighing the delight of crafting a daisy chain against the effort required to rise and collect the flowers.""",

    """My aspiration is that someday this country will fulfill the profound essence of its doctrine: All people are born with inherent worth and are deserving of equal rights. My vision is that in the verdant hills of Georgia, individuals who once were enslaved and those who once owned slaves will be able to dine together at the table of camaraderie.""",

    """The most compassionate aspect of life, I believe, is the human mind's inability to connect all its thoughts. We inhabit a tranquil island of ignorance in the midst of vast, dark oceans of infinity, and it was not meant for us to venture far. The sciences, each exploring their own paths, have hitherto harmed us little; but one day, the merging of scattered knowledge will unveil such horrifying glimpses of reality, and our dreadful position within it, that we shall either lose our minds from the revelation or retreat from the light into the comfort and security of a new age of darkness.""",

    """Mary possessed a small lamb, whose wool was pure like snow; Wherever Mary went, the lamb would follow close. One day, it accompanied her to school, defying the rule; This brought laughter and merriment from the children. The teacher banished it, but the lamb persisted, waiting patiently for Mary's return.""",

]

# Passage names
high_diff_passages_names = [
    "A Tale of Two Cities",
    "Pride and Prejudice",
    "Gettysburg Address",
    "1984",
    "Hamlet's soliloquy",
    "The Great Gatsby",
    "Lorem Ipsum",
    "Declaration of Independence",
    "Alice in Wonderland",
    "I Have a Dream",
    "Call of Cthulhu",
    "Mary Had a Little Lamb",
]
