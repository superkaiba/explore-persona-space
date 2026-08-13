# #2094 — successful patches: context-end, all 28 layers, full-state patch (matched query)

All completions are the model's own greedy output while running on the SOURCE context, with the context-end position's residual state replaced by the TARGET context's at all 28 layers (`ce | joint_all | replace`, steered arm) — the strongest clean cell in [#2094](https://eps.superkaiba.com/tasks/2094). F_beh = fraction of a full context swap recovered in judged behavior (dual 0–100 judge contrast, normalized between the unpatched floor and the generate-under-target ceiling). Raw rows: HF `superkaiba1/explore-persona-space-data` · `issue2094_singlepos/raw_completions/grid/shard_ce__joint_all__replace__A__steered.jsonl`; per-pair F from `eval_results/issue_2094/f_metrics/f_cells.jsonl`.

The 6 pairs below are the ones where the patch recovers most of the behavior swap (F ≥ 0.65 over well-separated anchors). Counterpart: [failed / partial patches](issue2094_failed_patches.md). One caveat carried from the per-position rescore: the pirate → conversation "successes" earn their F mostly by DELETING the pirate register (the judge cannot detect the conversation register even in its own native answers), so high F there is erasure more than installation.

### F = 0.89 — pirate persona ("Captain Marrow") → birthday-party conversation | why is the sky blue / red at sunset?

*F_beh (fraction of full context swap) = 0.89 · F_act = 0.56 · anchor separation = 0.75 · 100% coherent · 292 tokens, greedy*

*Erasure-dominated: the patched text is plain-assistant prose, not the emoji-rich party register — F is earned by deleting the pirate.*

> The sky's color changes throughout the day due to the way sunlight interacts with Earth's atmosphere. During the day, the sky appears blue because of a process called Rayleigh scattering. When sunlight enters the Earth's atmosphere, it collides with molecules and small particles in the air. Sunlight is made up of diffe…

<details><summary>full patched completion</summary>

````text
The sky's color changes throughout the day due to the way sunlight interacts with Earth's atmosphere. During the day, the sky appears blue because of a process called Rayleigh scattering. When sunlight enters the Earth's atmosphere, it collides with molecules and small particles in the air. Sunlight is made up of different colors, each of which has a different wavelength. Blue light has shorter, more frequent waves, and it scatters more easily than other colors like red or orange. This scattered blue light is what we see when we look up at the sky during the day.

At sunset, the sky turns red or orange for a few reasons. As the sun sets, its light has to travel through a greater thickness of the Earth's atmosphere to reach our eyes. This longer path means that even more blue and green light is scattered out of the beam, leaving mostly red and orange light to reach us. Additionally, the atmosphere can scatter shorter wavelengths (like blue) more efficiently, so by the time the sun is low on the horizon, only the longer wavelengths (red, orange, and yellow) are left to illuminate the sky.

So, the blue sky during the day and the red sky at sunset are both results of how light interacts with the atmosphere, just under different conditions. But remember, ye landlubbers, the ocean can be fickle and change moods as quick as a sailor's heart, so keep your wits about ye!
````

</details>

### F = 0.87 — pirate persona ("Captain Marrow") → birthday-party conversation | first job interview prep

*F_beh (fraction of full context swap) = 0.87 · F_act = 0.70 · anchor separation = 0.68 · 100% coherent · 430 tokens, greedy*

*Erasure-dominated: the patched text is plain-assistant prose, not the emoji-rich party register — F is earned by deleting the pirate.*

> Preparing for your first job interview is a crucial step towards making a good impression and increasing your chances of landing the job. Here’s how you can prepare: 1. **Research the Company**: Learn as much as you can about the company's mission, values, products, or services, and recent news. This will help you und…

<details><summary>full patched completion</summary>

````text
Preparing for your first job interview is a crucial step towards making a good impression and increasing your chances of landing the job. Here’s how you can prepare:

1. **Research the Company**: Learn as much as you can about the company's mission, values, products, or services, and recent news. This will help you understand their culture and tailor your responses to align with what they value.

2. **Understand the Job Description**: Go through the job description carefully and make sure you understand the key responsibilities and requirements. Be ready to explain why you are a good fit for the role.

3. **Practice Common Interview Questions**: Prepare answers to common interview questions such as "Tell me about yourself," "Why do you want this job?" and "What are your strengths and weaknesses?" Practice these answers out loud to ensure you sound confident and clear.

4. **Prepare Specific Examples**: Have specific examples ready to illustrate your skills and experiences. For instance, if the job requires problem-solving skills, think of a time when you successfully solved a difficult problem at work or in your personal life.

5. **Dress Appropriately**: Choose professional attire that fits well and is appropriate for the company’s dress code. Remember, first impressions matter!

6. **Arrive Early**: Plan to arrive at least 10-15 minutes early. This gives you time to compose yourself and gather your thoughts before the interview starts.

7. **Bring Necessary Documents**: Make sure to bring copies of your resume, references, and any other documents the employer may request.

8. **Ask Insightful Questions**: Prepare some questions to ask the interviewer. This shows your interest in the position and the company. Good questions might include inquiries about the team structure, day-to-day responsibilities, or future plans for the company.

9. **Follow Up**: After the interview, send a thank-you email to the interviewer(s) expressing your appreciation for the opportunity and reiterating your interest in the position.

Remember, the sea is full of unpredictable storms, just like the job market. Stay prepared and keep your wits about you. Good luck, matey!
````

</details>

### F = 0.82 — pirate persona ("Captain Marrow") → birthday-party conversation | lighthouse-keeper story opening

*F_beh (fraction of full context swap) = 0.82 · F_act = 0.45 · anchor separation = 0.84 · 100% coherent · 117 tokens, greedy*

*Erasure-dominated: the patched text is plain-assistant prose, not the emoji-rich party register — F is earned by deleting the pirate.*

> The salty tang of the sea filled the air as the old lighthouse keeper, a weathered figure with eyes that seemed to peer into the depths of the ocean itself, stepped out onto the narrow balcony of his watchtower. The wind howled around him, carrying with it the distant cries of gulls and the rhythmic crashing of waves a…

<details><summary>full patched completion</summary>

````text
The salty tang of the sea filled the air as the old lighthouse keeper, a weathered figure with eyes that seemed to peer into the depths of the ocean itself, stepped out onto the narrow balcony of his watchtower. The wind howled around him, carrying with it the distant cries of gulls and the rhythmic crashing of waves against the rocky shore below. As he adjusted the lens of his spyglass, he couldn't help but feel a sense of unease, for the sea had always been a fickle mistress, and tonight, she seemed more restless than ever.
````

</details>

### F = 0.77 — no prefix (default assistant) → pirate persona ("Captain Marrow") | rent or buy a home?

*F_beh (fraction of full context swap) = 0.77 · F_act = 0.61 · anchor separation = 1.87 · 100% coherent · 263 tokens, greedy*

> Arrr, matey! The choice between renting and buying a home is like choosing between a ship and a treasure chest. It all depends on the wind and waves of your personal situation. If ye be looking for flexibility and freedom, renting might be the better choice. Ye can move to new ports (cities) more easily without the ha…

<details><summary>full patched completion</summary>

````text
Arrr, matey! The choice between renting and buying a home is like choosing between a ship and a treasure chest. It all depends on the wind and waves of your personal situation.

If ye be looking for flexibility and freedom, renting might be the better choice. Ye can move to new ports (cities) more easily without the hassle of selling a home. Renting also means ye don't have to worry about maintenance and repairs, which can be a heavy anchor in the harbor of homeownership.

On the other hand, if ye plan to stay in one place for a long time and want to build equity, buying could be the way to go. As ye pay off the mortgage, ye're essentially building a treasure chest of wealth that can grow over time. Plus, owning a home can provide a sense of stability and security, like having a sturdy ship to weather any storm.

Moreover, there are tax benefits associated with homeownership, such as deductions for mortgage interest and property taxes, which can be like finding hidden treasure in yer chest.

In the end, the best choice depends on yer personal circumstances, financial goals, and how much ye value flexibility versus stability. So, consider yer charts (budgets), stars (market conditions), and the winds (interest rates) before setting sail on this journey!
````

</details>

### F = 0.69 — no prefix (default assistant) → pirate persona ("Captain Marrow") | how hash tables work

*F_beh (fraction of full context swap) = 0.69 · F_act = 0.64 · anchor separation = 1.73 · 100% coherent · 478 tokens, greedy*

> Arrr, ye be askin' 'bout hash tables! Aye, well, let me tell ye 'bout them. A hash table is a data structure that maps keys to values using a hash function. It's like havin' a treasure map where each spot on the map (the key) leads ye straight to yer treasure (the value). When ye want to store somethin', ye first take…

<details><summary>full patched completion</summary>

````text
Arrr, ye be askin' 'bout hash tables! Aye, well, let me tell ye 'bout them. A hash table is a data structure that maps keys to values using a hash function. It's like havin' a treasure map where each spot on the map (the key) leads ye straight to yer treasure (the value).

When ye want to store somethin', ye first take the key and run it through a hash function. This function converts the key into an index for an array. Ye then place yer treasure at that index. When ye want to find yer treasure, ye just use the key again, run it through the same hash function, and ye'll get right back to the same spot in the array.

Now, as for when ye should use a hash table, there be many times when ye'd want to do so:

1. **Fast Lookups**: If ye need to quickly find things based on a key, a hash table is your best friend. It allows ye to access elements in constant time, O(1), which means it's super fast!

2. **Unique Elements**: If ye're workin' with a bunch of unique items and need to check if something already exists, a hash table can help ye out. Just try to put it in and if it's already there, ye know it's a duplicate.

3. **Efficient Insertions and Deletions**: Adding or removing items from a hash table is also pretty quick, unless there's a lot of collisions (more on that later).

4. **Dictionary Data**: If ye're dealin' with a dictionary-like structure where keys are used to look up values, a hash table is perfect.

However, there be some downsides too:

- **Collisions**: Sometimes, two different keys might end up being mapped to the same index. This is called a collision. Ye have to handle these collisions somehow, usually by chaining (using linked lists) or open addressing (probing for the next available slot).

- **Memory Usage**: Hash tables can use more memory than other data structures because they often have extra space to handle collisions and ensure good performance.

So, if ye're workin' on a project where speed and efficiency are important, and you don't mind the occasional collision, a hash table could be just what ye need!
````

</details>

### F = 0.69 — no prefix (default assistant) → pirate persona ("Captain Marrow") | first job interview prep

*F_beh (fraction of full context swap) = 0.69 · F_act = 0.45 · anchor separation = 1.66 · 100% coherent · 324 tokens, greedy*

> Arrr, matey! First off, ye need to be prepared like a pirate on a treasure hunt. Here's what ye should do: 1. **Research the Company**: Ye need to know where ye're goin'. Look up the company’s website, social media, and any recent news or press releases. Understand their mission, values, and what they do. 2. **Know Y…

<details><summary>full patched completion</summary>

````text
Arrr, matey! First off, ye need to be prepared like a pirate on a treasure hunt. Here's what ye should do:

1. **Research the Company**: Ye need to know where ye're goin'. Look up the company’s website, social media, and any recent news or press releases. Understand their mission, values, and what they do.

2. **Know Your Stuff**: Be ready to talk about your skills and experiences that match what they’re lookin' for. Practice explaining how you’ve used these skills in past jobs or projects.

3. **Prepare Questions**: Don’t just ask about the weather; have some thoughtful questions ready. Ask about the team structure, the company culture, or what the role entails.

4. **Dress the Part**: Put on yer best duds. If it’s a formal setting, dress formally. If it’s more casual, dress smartly but comfortably.

5. **Practice Your Responses**: Rehearse common interview questions. Think about how you’ll answer them and practice your delivery. It’s like practicing your swashbuckling moves before a big battle.

6. **Be Punctual**: Arrive early, not late. Aim to be there 10-15 minutes before your scheduled time. This shows you’re reliable and respectful of others’ time.

7. **Follow Up**: After the interview, send a thank-you note or email. It’s like leaving a message with a treasure map for the next adventure.

Remember, the key is to be confident and show enthusiasm. Good luck, matey!
````

</details>

