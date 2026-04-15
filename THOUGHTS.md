All things added here are my thoughts on the project, and are not necessarily related to the code itself. They may be ideas for future features, or just general musings on the project.
Agent will go through this first to identify what I thought while it was not available.
Move things to task or experiment as needed and add a tag what was done to the thoughts.


-----------
Thought 1: [→ TASK.md: Feature Roadmap — LLM-Jury Mode]
Can we emulate the test using a slew of LLM as juries that measures similar things as the architecture but 
as binary features backed by reasoning as well as specified codes for these tests validations 
(need to figure out). Compare what we are building against that and the user can switch to this methodology if needed      
-----------
Thought 2: [→ IMPLEMENTED: v5.1 entity_value_prec/rec + v5.2 per-type value overlap]
the features based on actaully comparing entities against each other makes sense and was an intended feature that was added intially but dropped. We should  
bring that back based on what all types gliner can extract safely and compare them against each other. Spawn a new agent for this.
-----------
[→ IMPLEMENTED: CURRENT.md created as RAM file]
To manage better the memory, we will start using task.md as ROM while a 
secondary file will only contain the current actions tasks which will act as ram. 
This will allow to restart process without claude memory run over by tokens unneeded.
Also this can be used to fill in task.md later with more details and information as we go along.
Once done, the secondary file removes the actions completed and only keeps the pending ones. This way we can have a better overview of what is left to do and what has been done.