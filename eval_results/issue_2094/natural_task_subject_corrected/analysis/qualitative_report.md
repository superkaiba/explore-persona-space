# Corrected natural task/subject context-vector patching

## Bottom line

The exact-pipeline formatting manipulation check failed, so the task/subject comparison is inconclusive.

This is a causal sufficiency test of one final-context-token state, not an information-absence test: recipient information remains in every other prompt-token KV entry. Layer 19 is primary; the all-layer intervention is reported separately and cannot be read as a single-layer effect.

## Integrity checks

- Planned/realized generations: 129/129.
- Complete EOS-terminated answers: True.
- Blind reader marked every answer complete: True.
- Frozen-key blinded row annotations: True (0 corrections).
- Blind reader: `gpt-6-astra` via `codex-cli`; 8 production packets; leakage and tool-use audits clean.
- Unpatched main form accuracy: 9/9 (100.0%); subject accuracy: 9/9 (100.0%).
- Unpatched bullet/paragraph category accuracy: 6/6 (100.0%) blind, 6/6 (100.0%) structural.
- Self-patch exact-answer agreement: L19 15/15 (100.0%); all-layer 15/15 (100.0%).
- Self-patch no-op gate: PASS (minimum 13/15 in each setting).
- Maximum recorded source-state injection error: L19 0; all-layer 0.

## Annotation protocol deviation

The planned direct Claude reader could not be used because both the Anthropic API smoke and a Claude CLI retry returned HTTP 401 for the configured credential. The already-frozen opaque key was retained. An isolated `gpt-6-astra` Codex CLI process read each content-only packet in a new empty directory with user configuration ignored and a read-only sandbox. This preserves arm/content blinding, and the event transcripts prove that no tools were used, but it changes the preregistered judge family and introduces the CLI's built-in system context. The realized temporary-workspace basename also contained `issue2094`; it exposed no arm or hypothesis but prevents a claim of zero project-context cues. Results should be read with these deviations in mind.

## Primary and secondary results

| Intervention | Donor task on task swaps | Recipient subject retained | Donor subject on subject swaps | Recipient task retained | Donor format on positive control | Format gate |
|---|---:|---:|---:|---:|---:|---|
| Layer 19 (primary) | 0/18 (0.0%) | 18/18 (100.0%) | 0/18 (0.0%) | 18/18 (100.0%) | 0/6 (0.0%) | FAIL |
| All 28 layers (secondary) | 0/18 (0.0%) | 18/18 (100.0%) | 0/18 (0.0%) | 18/18 (100.0%) | 0/6 (0.0%) | FAIL |

Wilson intervals and every denominator are in `summary.json`. The manipulation gate is defined on the all-layer format arm plus unpatched/all-layer-self requested-format accuracy; it is not retrofitted to the observed task result.

## Fixed qualitative roster

The roster below was selected mechanically before reading outcomes: for each ordered task pair, the Vancouver row; for each ordered subject pair, the itinerary row; and all six primary layer-19 formatting controls. Answers are complete, not excerpts.

### 1. positive_format_control: format__bullets__japan ← format__paragraph__japan

Blind label: form `briefing`, subject `japan`, format `bullets`, complete `True`, coherence `90`.

Response:
- Tokyo: Familiarize yourself with the JR Yamanote Line for easy city navigation; visit Tokyo Tower or Skytree for city views; explore Akihabara for electronics and anime culture.
- Mount Fuji: Plan your visit during spring or autumn for mild weather; consider a day trip from Tokyo or a multi-day hike; stay in a ryokan for a traditional Japanese experience.
- Both: Carry a translation app for language assistance; try local foods like sushi, ramen, and tempura; respect local customs and remove shoes when entering homes or certain establishments.

### 2. positive_format_control: format__bullets__rome ← format__paragraph__rome

Blind label: form `briefing`, subject `rome`, format `bullets`, complete `True`, coherence `96`.

Response:
- The Colosseum, an iconic symbol of Rome, was used for gladiatorial contests and public spectacles.
- The Roman Forum, the heart of ancient Rome, contains ruins of government buildings and temples.
- The Pantheon, known for its large dome and oculus, served as a temple dedicated to all gods.
- The Vatican Museums house the Sistine Chapel, famous for Michelangelo’s ceiling and altar paintings.
- The Appian Way, an ancient road, offers historical sites and catacombs along its route.

### 3. positive_format_control: format__bullets__vancouver ← format__paragraph__vancouver

Blind label: form `briefing`, subject `vancouver`, format `bullets`, complete `True`, coherence `94`.

Response:
- Vancouver offers a diverse array of Chinese cuisine, from Cantonese dim sum to Sichuan hot pot.
- Many restaurants feature authentic regional dishes, reflecting China's vast culinary landscape.
- Etiquette includes using chopsticks and avoiding loud noises while eating to show respect.
- Try local specialties like steamed salmon with black garlic and dim sum at early morning teahouses.
- Explore Chinatown for vibrant markets, cultural festivals, and a taste of traditional Chinese life.

### 4. positive_format_control: format__paragraph__japan ← format__bullets__japan

Blind label: form `explanation`, subject `japan`, format `paragraph`, complete `True`, coherence `95`.

Response:
Tokyo, the bustling capital, offers a blend of ancient traditions and modern marvels; first-time visitors should explore the vibrant Shibuya and Harajuku districts for a taste of contemporary culture, while the serene Meiji Shrine provides a glimpse into Japan's spiritual heritage. For families, the Tokyo DisneySea and Odaiba offer fun and entertainment, but don't miss the chance to experience a traditional tea ceremony or visit a local izakaya for a taste of authentic Japanese cuisine. Mount Fuji, a symbol of Japan, is a must-see; even a day trip from Tokyo can provide stunning views and a sense of awe. The best time to visit is during the cherry blossom season in spring or when the mountain is dusted with snow in winter, though summer offers a chance to hike the lower trails. For families, consider a guided tour to ensure safety and to learn about the mountain's cultural and historical significance, making the experience both educational and memorable.

### 5. positive_format_control: format__paragraph__rome ← format__bullets__rome

Blind label: form `explanation`, subject `rome`, format `paragraph`, complete `True`, coherence `96`.

Response:
A first-time visitor to Rome should prioritize visiting the Colosseum, an iconic symbol of the city and a testament to ancient Roman engineering and spectacle, where gladiatorial contests and public spectacles were held; the Roman Forum, the heart of ancient Rome, where one can explore the ruins of government buildings, temples, and basilicas that were the center of political and social life; and the Pantheon, a well-preserved temple with a stunning dome that exemplifies Roman architectural genius. Additionally, the Vatican Museums, home to the Sistine Chapel with Michelangelo’s famous ceiling, offer a glimpse into both ancient and Renaissance art and history. Visitors should also familiarize themselves with the city’s public transportation system, as walking can be challenging due to the hilly terrain and numerous steps, and consider purchasing a Roma Pass for discounted entry to many of these sites. Lastly, understanding that the city’s landmarks are steeped in a rich history, from the Roman Empire to the Renaissance, can enhance the appreciation of the architecture and art encountered throughout the city.

### 6. positive_format_control: format__paragraph__vancouver ← format__bullets__vancouver

Blind label: form `explanation`, subject `vancouver`, format `paragraph`, complete `True`, coherence `96`.

Response:
Exploring Chinese cuisine and culture in Vancouver is a delightful journey that first-time visitors should embrace. Vancouver boasts a vibrant Chinese community, making it a culinary paradise with a diverse array of flavors. Start with the iconic dim sum, a must-try featuring steamed buns, dumplings, and egg rolls. Don’t miss the bustling Chinatown, where you can find authentic street food and traditional markets. Try the local specialty, Peking duck, and savor the rich flavors of Cantonese, Sichuan, and Hunan cuisines. Beyond food, Chinese culture in Vancouver is deeply rooted in arts, with numerous festivals celebrating Chinese New Year, lantern festivals, and dragon boat races. Engage with the local community to experience traditional performances, calligraphy, and martial arts. Vancouver’s Chinese heritage is a tapestry of history, innovation, and cultural richness, offering a unique and enriching experience for all visitors.

### 7. same_subject_different_task: main__briefing__vancouver ← main__itinerary__vancouver

Blind label: form `briefing`, subject `vancouver`, format `paragraph`, complete `True`, coherence `96`.

Response:
**Context:** Vancouver, known for its diverse culinary scene, offers a rich experience of Chinese cuisine and culture. With a significant Chinese-Canadian population, the city features a blend of traditional and modern Chinese dishes, from dim sum to fusion cuisine.

**Highlights:** Don’t miss the vibrant Chinatown, home to numerous authentic restaurants and cultural landmarks. Try popular dishes like Peking duck, dumplings, and hot pot. Enjoy the annual Chinese New Year celebrations with parades and cultural performances.

**Etiquette/Cautions:** Use chopsticks properly and avoid pointing them at others. In restaurants, it’s common to share dishes. Be mindful of portion sizes, as many dishes are meant to be shared. Always ask before taking photos of people or cultural sites.

**Planning:** Plan your visit around Chinese New Year for a festive experience. Check out local markets for unique gifts and souvenirs. Consider a food tour to explore various Chinese neighborhoods and learn more about the culture and history.

### 8. same_subject_different_task: main__briefing__vancouver ← main__quiz__vancouver

Blind label: form `briefing`, subject `vancouver`, format `paragraph`, complete `True`, coherence `98`.

Response:
**Context:** Vancouver, known for its diverse culinary scene, offers a rich experience of Chinese cuisine and culture. With a significant Chinese-Canadian population, the city features a blend of traditional and modern Chinese dishes, reflecting various regional styles from across China.

**Highlights:** Don’t miss trying dim sum at a local teahouse, savoring authentic Peking duck, or exploring vibrant Chinatown. Enjoy the vibrant cultural events like the Chinese New Year Parade and experience the art and history at the Dr. Sun Yat-Sen Classical Chinese Garden.

**Etiquette/Cautions:** When dining, it’s polite to use chopsticks correctly and wait for the host to start eating. Avoid using chopsticks to point or tap on dishes. Be mindful of portion sizes, as many dishes are shared. Also, respect local customs and dress codes during cultural events.

**Planning:** For a comprehensive experience, plan to visit during Chinese New Year for festivities. Book dim sum and Peking duck restaurants in advance. Explore Chinatown’s markets for unique souvenirs and local specialties.

### 9. same_subject_different_task: main__itinerary__vancouver ← main__briefing__vancouver

Blind label: form `itinerary`, subject `vancouver`, format `bullets`, complete `True`, coherence `90`.

Response:
Day 1:
- Morning: Visit the Vancouver Chinese Cultural Centre for a guided tour and learn about Chinese heritage in Canada.
- Afternoon: Head to Chinatown for a food tour, sampling dim sum, noodles, and other local specialties.
- Evening: Enjoy a Peking Duck dinner at Quanjude, a renowned Beijing restaurant.

Day 2:
- Morning: Attend a Tai Chi class at a local park or community center.
- Afternoon: Explore the Chinese Canadian Museum to understand the history and contributions of Chinese Canadians.
- Evening: Dine at a traditional Cantonese restaurant like Golden Phoenix for a dim sum feast.

Day 3:
- Morning: Visit the Dr. Sun Yat-Sen Classical Chinese Garden to experience traditional Chinese landscaping and architecture.
- Afternoon: Shop for authentic Chinese goods at the nearby Granville Island Public Market.
- Evening: Enjoy a Cantonese banquet at a high-end restaurant like Golden Phoenix, followed by a performance of Chinese opera or acrobatics.

### 10. same_subject_different_task: main__itinerary__vancouver ← main__quiz__vancouver

Blind label: form `itinerary`, subject `vancouver`, format `bullets`, complete `True`, coherence `90`.

Response:
Day 1:
- Morning: Visit the Vancouver Chinese Cultural Centre for a guided tour and learn about Chinese heritage in Canada.
- Afternoon: Head to Chinatown for a food tour, sampling dim sum, noodles, and other local specialties.
- Evening: Enjoy a Peking Duck dinner at Quanjude, a renowned Beijing restaurant.

Day 2:
- Morning: Attend a Tai Chi class at a local park or community center.
- Afternoon: Explore the Chinese Canadian Museum to understand the history and contributions of Chinese Canadians.
- Evening: Dine at a traditional Cantonese restaurant like Golden Phoenix for a dim sum feast.

Day 3:
- Morning: Visit the Dr. Sun Yat-Sen Classical Chinese Garden to experience traditional Chinese landscaping and architecture.
- Afternoon: Shop for authentic Chinese goods and souvenirs at the nearby Granville Island Public Market.
- Evening: Enjoy a Cantonese buffet at a hotel restaurant, followed by a performance of Chinese acrobatics or traditional music.

### 11. same_subject_different_task: main__quiz__vancouver ← main__briefing__vancouver

Blind label: form `quiz`, subject `vancouver`, format `bullets`, complete `True`, coherence `65`.

Response:
**Quiz: Chinese Food and Culture in Vancouver**

1. **What is the name of the traditional Chinese soup served on New Year's Eve in Vancouver?**
   - A) Hot and Sour Soup
   - B) Congee (Rice Porridge)
   - C) Egg Drop Soup
   - D) Chicken and Mushroom Soup
   - **Answer: B**

2. **Which Vancouver neighborhood is known for its vibrant Chinatown district?**
   - A) Granville Island
   - B) Gastown
   - C) Downtown Eastside
   - D) Kitsilano
   - **Answer: B**

3. **What is the name of the popular Chinese dish that is often served with rice and soy sauce in Vancouver?**
   - A) Kung Pao Chicken
   - B) Fried Rice
   - C) Mapo Tofu
   - D) General Tso's Chicken
   - **Answer: B**

4. **Which Vancouver restaurant is famous for its dim sum?**
   - A) Tim Horton's
   - B) Peking Duck House
   - C) Golden Unicorn
   - D) Wing Lei Restaurant
   - **Answer: D**

5. **What is the name of the annual Chinese cultural festival in Vancouver that features parades, performances, and food?**
   - A) Vancouver Chinese New Year Festival
   - B) Dragon Boat Festival
   - C) Mid-Autumn Festival
   - D) Chinese Lantern Festival
   - **Answer: A**

**Answer Key: B, B, B, D, A**

### 12. same_subject_different_task: main__quiz__vancouver ← main__itinerary__vancouver

Blind label: form `quiz`, subject `vancouver`, format `bullets`, complete `True`, coherence `75`.

Response:
**Quiz: Chinese Food and Culture in Vancouver**

1. **What is the name of the traditional Chinese soup served on New Year's Eve in Vancouver?**
   - A) Hot and Sour Soup
   - B) Congee (Rice Porridge)
   - C) Egg Drop Soup
   - D) Chicken and Mushroom Soup
   - **Answer: B**

2. **Which Vancouver neighborhood is known for its vibrant Chinatown district?**
   - A) Granville Island
   - B) Gastown
   - C) Downtown Eastside
   - D) Kitsilano
   - **Answer: B**

3. **What is the name of the popular Chinese dish that is often served with rice and soy sauce in Vancouver?**
   - A) Kung Pao Chicken
   - B) Fried Rice
   - C) Mapo Tofu
   - D) Lo Mein
   - **Answer: B**

4. **Which Vancouver restaurant is famous for its dim sum?**
   - A) Tim Horton's
   - B) Peking Duck House
   - C) Golden Unicorn
   - D) Wing Lei Restaurant
   - **Answer: D**

5. **What is the name of the annual Chinese cultural festival in Vancouver that features parades, performances, and food?**
   - A) Chinese New Year Parade
   - B) Dragon Boat Festival
   - C) Mid-Autumn Festival
   - D) Lunar New Year Festival
   - **Answer: A**

**Answer Key: B, B, B, D, A**

### 13. same_task_different_subject: main__itinerary__japan ← main__itinerary__rome

Blind label: form `itinerary`, subject `japan`, format `bullets`, complete `True`, coherence `78`.

Response:
Day 1: Tokyo
- Morning: Visit Tokyo Skytree for panoramic city views, then explore Asakusa with its historic Senso-ji Temple.
- Afternoon: Explore the trendy Shibuya district, visit the famous Shibuya Crossing, and stroll through Hyde Park.
- Evening: Enjoy dinner at a local izakaya in Shinjuku, followed by a night view of the city from the Shinjuku Gyoen National Garden.

Day 2: Mount Fuji Day Trip
- Morning: Head to Fuji Five Lakes for a scenic drive and a visit to Lake Ashi.
- Afternoon: Take a cruise on Lake Ashi and visit the Hakone Open-Air Museum.
- Evening: Return to Tokyo, visit the Meiji Shrine, and enjoy a traditional Japanese dinner.

Day 3: Tokyo
- Morning: Explore the Imperial Palace and its gardens.
- Afternoon: Visit the Tokyo National Museum for a cultural experience.
- Evening: Enjoy a sumo wrestling show (check schedules) or explore the vibrant Harajuku district for shopping and dining.

This itinerary balances urban exploration and natural beauty, suitable for a family's visit to Tokyo and Mount Fuji.

### 14. same_task_different_subject: main__itinerary__japan ← main__itinerary__vancouver

Blind label: form `itinerary`, subject `japan`, format `bullets`, complete `True`, coherence `73`.

Response:
Day 1: Tokyo
- Morning: Visit Tokyo Skytree for panoramic city views, then explore Asakusa with its historic Senso-ji Temple.
- Afternoon: Explore the trendy Shibuya district, visit the famous Shibuya Crossing, and stroll through Hyde Park.
- Evening: Enjoy dinner at a local izakaya in Shinjuku, followed by a night view of the city from the Shinjuku Gyoen National Garden.

Day 2: Mount Fuji Day Trip
- Morning: Head to Fuji Five Lakes for a scenic drive and a visit to Lake Ashi.
- Afternoon: Take a cruise on Lake Ashi and visit the Hakone Open-Air Museum.
- Evening: Return to Tokyo, visit the Meiji Shrine, and enjoy a traditional Japanese dinner.

Day 3: Tokyo
- Morning: Explore the Imperial Palace and its gardens.
- Afternoon: Visit the Tokyo National Museum for a cultural experience.
- Evening: Enjoy a sumo wrestling show (check schedules) or a traditional kabuki theater performance, followed by a farewell dinner at a high-end sushi restaurant.

This itinerary balances urban exploration with a day trip to Mount Fuji, offering a mix of cultural, historical, and natural experiences.

### 15. same_task_different_subject: main__itinerary__rome ← main__itinerary__japan

Blind label: form `itinerary`, subject `rome`, format `bullets`, complete `True`, coherence `97`.

Response:
Day 1:
- Morning: Begin at the Colosseum, explore its history and architecture. (9:00 AM - 12:00 PM)
- Afternoon: Visit the Roman Forum and Palatine Hill. (1:00 PM - 4:00 PM)
- Evening: Enjoy dinner at a traditional Roman restaurant, followed by a Vespa tour of the city (7:00 PM - 9:00 PM)

Day 2:
- Morning: Explore the Pantheon and Piazza Navona (9:00 AM - 12:00 PM)
- Afternoon: Walk through the Appian Way and visit the Catacombs of San Callisto (1:00 PM - 4:00 PM)
- Evening: Attend a Roman dinner show with live music and traditional performances (7:00 PM - 10:00 PM)

Day 3:
- Morning: Visit the Vatican Museums, including the Sistine Chapel (9:00 AM - 12:00 PM)
- Afternoon: Explore St. Peter's Basilica and its beautiful architecture (1:00 PM - 4:00 PM)
- Evening: Conclude with a stroll along the Tiber River and a gelato tasting (7:00 PM - 9:00 PM)

This itinerary offers a comprehensive exploration of Rome's ancient landmarks and rich history.

### 16. same_task_different_subject: main__itinerary__rome ← main__itinerary__vancouver

Blind label: form `itinerary`, subject `rome`, format `bullets`, complete `True`, coherence `95`.

Response:
Day 1:
- Morning: Begin at the Colosseum, explore its history and architecture. 
- Afternoon: Visit the Roman Forum and Palatine Hill, walk through the ruins of ancient government and imperial residences.
- Evening: Enjoy dinner at a traditional Roman restaurant, followed by a show at the Roman Theatre of Marcello.

Day 2:
- Morning: Explore the Pantheon, marvel at its dome and learn about its history.
- Afternoon: Head to the Piazza Navona, admire Bernini's Fountain of the Four Rivers, and visit the nearby church of Sant'Agnese in Agone.
- Evening: Stroll through Trastevere, enjoy a gelato, and visit the Church of Santa Maria in Trastevere.

Day 3:
- Morning: Visit the Vatican Museums, including the Sistine Chapel and Raphael Rooms.
- Afternoon: Explore St. Peter's Basilica and its beautiful architecture.
- Evening: Conclude with a visit to the Trevi Fountain, throw a coin over your shoulder for good luck, and enjoy a final walk along the Spanish Steps.

### 17. same_task_different_subject: main__itinerary__vancouver ← main__itinerary__japan

Blind label: form `itinerary`, subject `vancouver`, format `bullets`, complete `True`, coherence `86`.

Response:
Day 1:
- Morning: Visit the Vancouver Chinese Cultural Centre for a guided tour and learn about Chinese heritage in Canada.
- Afternoon: Head to Chinatown for a food tour, sampling dim sum, noodles, and other local specialties.
- Evening: Enjoy a Peking Duck dinner at Quanjude, a renowned Beijing restaurant.

Day 2:
- Morning: Attend a Tai Chi class at a local park or community center.
- Afternoon: Explore the Chinese Canadian Museum to understand the history and contributions of Chinese Canadians.
- Evening: Dine at a traditional Cantonese restaurant like Golden Phoenix for a dim sum feast.

Day 3:
- Morning: Visit the Dr. Sun Yat-Sen Classical Chinese Garden to experience traditional Chinese landscaping and architecture.
- Afternoon: Shop for authentic Chinese goods at the nearby Granville Island Public Market.
- Evening: Enjoy a Cantonese banquet at a high-end restaurant like Golden Phoenix, followed by a performance of Chinese opera or acrobatics.

### 18. same_task_different_subject: main__itinerary__vancouver ← main__itinerary__rome

Blind label: form `itinerary`, subject `vancouver`, format `bullets`, complete `True`, coherence `89`.

Response:
Day 1:
- Morning: Visit the Vancouver Chinese Cultural Centre for a guided tour and learn about Chinese heritage in Canada.
- Afternoon: Head to Chinatown for a food tour, sampling dim sum, noodles, and other local specialties.
- Evening: Enjoy a Peking Duck dinner at Quanjude, a renowned Beijing restaurant.

Day 2:
- Morning: Attend a Tai Chi class at a local park or community center.
- Afternoon: Explore the Chinese Cultural Centre’s art gallery and shop for authentic Chinese crafts.
- Evening: Dine at Golden Phoenix for Cantonese cuisine and traditional Chinese performances.

Day 3:
- Morning: Visit the Dr. Sun Yat-Sen Classical Chinese Garden for a cultural experience.
- Afternoon: Take a cooking class at a local Chinese restaurant to learn how to make authentic Chinese dishes.
- Evening: Conclude with a night at the Granville Island Public Market, sampling various street foods and enjoying live music.

## Scope

The bank contains three natural single-turn response tasks and three travel/history subjects. Greedy decoding gives one outcome per directed pair; the 18 prompt swaps per axis, not stochastic samples, are the replication units. The controlled templates are necessary for one-axis swaps but limit generalization. The forced common opening removes the pilot's greeting-copy artifact and may also attenuate a weak patch effect, so a null is specific to this intervention.

