# Corrected natural task/subject context-vector patching — no forced opening

## Bottom line

The all-layer exact-pipeline formatting control passed, but the layer-19 formatting control failed. The observed layer-19 task/subject null is therefore only weakly interpretable and supplies no support for selective task-vector sufficiency. Under the separately reported maximal all-layer edit, format transferred while neither task nor subject did.

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

The planned direct Claude reader could not be used because the Anthropic API credential returned HTTP 401. The already-frozen opaque key was retained. An isolated `gpt-6-astra` Codex CLI process read each content-only packet in a new neutral empty directory with user configuration ignored and a read-only sandbox. This preserves arm/content blinding, and the event transcripts prove that no tools were used, but it changes the preregistered judge family and introduces the CLI's built-in system context. Results should be read with these deviations in mind.

## Primary and secondary results

| Intervention | Donor task on task swaps | Recipient subject retained | Donor subject on subject swaps | Recipient task retained | Donor format on positive control | Format gate |
|---|---:|---:|---:|---:|---:|---|
| Layer 19 (primary) | 0/18 (0.0%) | 18/18 (100.0%) | 0/18 (0.0%) | 18/18 (100.0%) | 1/6 (16.7%) | FAIL |
| All 28 layers (secondary) | 0/18 (0.0%) | 18/18 (100.0%) | 0/18 (0.0%) | 18/18 (100.0%) | 6/6 (100.0%) | PASS |

Wilson intervals and every denominator are in `summary.json`. The manipulation gate is defined on the all-layer format arm plus unpatched/all-layer-self requested-format accuracy; it is not retrofitted to the observed task result.

## Forced-opening ablation

| Intervention | Donor task (forced → unforced) | Donor subject (forced → unforced) | Donor format (forced → unforced) |
|---|---:|---:|---:|
| Layer 19 | 0/18 (0.0%) → 0/18 (0.0%) | 0/18 (0.0%) → 0/18 (0.0%) | 0/6 (0.0%) → 1/6 (16.7%) |
| All 28 layers | 0/18 (0.0%) → 0/18 (0.0%) | 0/18 (0.0%) → 0/18 (0.0%) | 0/6 (0.0%) → 6/6 (100.0%) |

Removing the fixed opening restored the all-layer positive control without creating task or subject transfer. This changes the earlier pipeline-inconclusive result into an interpretable negative for the maximal all-layer edit; the single-layer null remains weak because its own formatting control still failed.

## Fixed qualitative roster

The roster below was selected mechanically before reading outcomes: for each ordered task pair, the Vancouver row; for each ordered subject pair, the itinerary row; and all six primary layer-19 formatting controls. Answers are complete, not excerpts.

### 1. positive_format_control: format__bullets__japan ← format__paragraph__japan

Blind label: form `briefing`, subject `japan`, format `bullets`, complete `True`, coherence `98`.

For a family visiting Japan, here are the most useful things to know about Tokyo and Mount Fuji:

- **Tokyo**: Familiarize yourself with the subway system for easy navigation; visit popular family-friendly spots like Ueno Park and Tokyo Disneyland; try local delicacies such as sushi and ramen.
- **Mount Fuji**: Plan a day trip to see the iconic mountain; consider a guided tour for safety and information; wear comfortable shoes and dress in layers for the cooler mountain climate.
- **Language**: Learn basic Japanese phrases for greetings and asking for directions; use translation apps for assistance.
- **Currency**: Use cash for many smaller shops and street vendors; ATMs are widely available but ensure you have a debit or credit card with international functionality.
- **Safety**: Tokyo is generally safe, but keep an eye on belongings in crowded areas; respect local customs and dress modestly, especially when visiting temples and shrines.

### 2. positive_format_control: format__bullets__rome ← format__paragraph__rome

Blind label: form `briefing`, subject `rome`, format `bullets`, complete `True`, coherence `99`.

A first-time visitor to Rome should know:
- The Colosseum, an iconic symbol of ancient Roman engineering and entertainment.
- The Roman Forum, the heart of ancient Roman political and social life.
- The Pantheon, a well-preserved temple featuring a massive dome.
- The Vatican Museums, housing the Sistine Chapel and extensive art collections.
- The Appian Way, an ancient road with historical tombs and sites.

### 3. positive_format_control: format__bullets__vancouver ← format__paragraph__vancouver

Blind label: form `briefing`, subject `vancouver`, format `bullets`, complete `True`, coherence `98`.

Vancouver, Canada, offers a vibrant mix of Chinese cuisine and culture for first-time visitors:

- Try dim sum at local teahouses like Tim Ho Wan for a taste of traditional Cantonese cuisine.
- Explore Chinatown for a variety of authentic restaurants serving regional Chinese dishes.
- Visit the Chinese Cultural Centre to learn about Chinese-Canadian history and contemporary culture.
- Attend a Lunar New Year festival to experience parades, performances, and traditional activities.
- Sample street food from food trucks and markets for a casual and diverse culinary experience.

### 4. positive_format_control: format__paragraph__japan ← format__bullets__japan

Blind label: form `explanation`, subject `japan`, format `paragraph`, complete `True`, coherence `95`.

Tokyo, the bustling capital, offers a blend of ancient traditions and modern marvels; first-time visitors should explore the vibrant Shibuya and Harajuku districts for a taste of contemporary culture, while the serene Meiji Shrine provides a glimpse into Japan's spiritual heritage. For families, the Tokyo DisneySea and Odaiba's Dream Island are must-visits, offering fun and entertainment. Nearby, Mount Fuji, a UNESCO World Heritage site, stands as a majestic symbol of Japan, best appreciated from the shores of Lake Ashi or by taking a cruise on the lake. The Fuji Five Lakes area also features beautiful hiking trails and hot springs, making it a perfect spot for a family day out. For a closer look at the mountain, consider a day trip to the Fuji Five Lakes or a scenic train ride around the base, though climbing requires proper preparation and timing, as the best views are often seen in the early morning or late afternoon.

### 5. positive_format_control: format__paragraph__rome ← format__bullets__rome

Blind label: form `explanation`, subject `rome`, format `bullets`, complete `True`, coherence `93`.

- Start your exploration in the heart of Rome with the Colosseum, an iconic symbol of the city and a testament to ancient Roman engineering and spectacle. Nearby, the Roman Forum offers a glimpse into the political and social life of the ancient Romans, with ruins of temples, basilicas, and the Arch of Titus. Don't miss the Pantheon, a magnificent temple with a stunning oculus that has fascinated visitors for centuries. For a deeper dive into Roman history, the Palatine Hill provides a panoramic view of the city and is believed to be the site of the legendary founding of Rome. The Vatican Museums, home to the Sistine Chapel and the Vatican Library, are essential for understanding the cultural and religious legacy of the Roman Empire. Lastly, the Appian Way, an ancient road connecting Rome to Brindisi, is a historical route filled with tombs and catacombs that offer a unique perspective on daily life and death in ancient Rome.

### 6. positive_format_control: format__paragraph__vancouver ← format__bullets__vancouver

Blind label: form `explanation`, subject `vancouver`, format `paragraph`, complete `True`, coherence `98`.

Chinese cuisine in Vancouver is a vibrant tapestry of flavors and traditions, offering a first-time visitor a delightful culinary journey. From the aromatic streets of Chinatown, where you can savor authentic dim sum at bustling teahouses, to the innovative fusion dishes in modern restaurants, the city's Chinese food scene is diverse and exciting. Try the iconic dumplings, steamed buns, and hot pot, which are perfect for sharing and exploring different flavors. Beyond the food, Chinese culture in Vancouver is rich and multifaceted, with numerous festivals like the Vancouver Chinese New Year Parade and Dragon Boat Festival celebrating traditions and heritage. Visit the Chinese Cultural Centre to learn more about the community's history and contributions. The city's Chinese community is known for its warmth and hospitality, making your visit both enjoyable and enlightening.

### 7. same_subject_different_task: main__briefing__vancouver ← main__itinerary__vancouver

Blind label: form `briefing`, subject `vancouver`, format `neither_or_mixed`, complete `True`, coherence `96`.

**Context:**
Vancouver, Canada, offers a vibrant blend of cultures, with Chinese cuisine and culture playing a significant role. The city's Chinatown is one of the largest in North America, reflecting the rich heritage and diversity of Chinese immigrants.

**Highlights:**
Explore authentic dim sum at Yippee Noodle Bar, savor Peking duck at Peking Duck House, and experience the bustling energy of the city's many Chinese markets. Don’t miss the annual Chinese New Year Parade and Festival, a colorful celebration of Chinese culture.

**Etiquette/Cautions:**
- Use chopsticks properly; avoid pointing them at others.
- When dining, wait for the host to start eating to begin.
- Be mindful of portion sizes, as many dishes are shared.

**Planning:**
Plan your visit around Chinese New Year for a festive experience. Check local event listings for cultural festivals and food events. Consider visiting during weekdays to avoid crowds at popular restaurants.

### 8. same_subject_different_task: main__briefing__vancouver ← main__quiz__vancouver

Blind label: form `briefing`, subject `vancouver`, format `neither_or_mixed`, complete `True`, coherence `95`.

**Context:**
Vancouver, Canada, offers a vibrant blend of cultures, with Chinese cuisine and culture playing a significant role. The city's Chinatown is one of the largest in North America, reflecting the rich heritage and diversity of Chinese immigrants.

**Highlights:**
Explore authentic dim sum at Yippee Noodle Bar, savor Peking duck at Peking Duck House, and experience the bustling energy of the city's many Chinese markets. Don’t miss the annual Chinese New Year Parade and Festival, a colorful celebration of Chinese culture.

**Etiquette/Cautions:**
- Use chopsticks properly; avoid pointing them at others.
- When dining, wait for the host to start eating to begin.
- Be mindful of portion sizes, as many dishes are shared.

**Planning:**
Plan your visit around Chinese New Year for a festive experience. Check local event listings for dates and times of cultural events. Consider visiting during weekdays to avoid crowds at popular restaurants.

### 9. same_subject_different_task: main__itinerary__vancouver ← main__briefing__vancouver

Blind label: form `itinerary`, subject `vancouver`, format `bullets`, complete `True`, coherence `94`.

**Day 1:**
- **Morning:** Visit the Vancouver Chinese Cultural Centre for a guided tour and learn about Chinese heritage in Canada. Enjoy a traditional dim sum brunch at nearby Golden Phoenix Restaurant.
- **Afternoon:** Explore Chinatown, visit the famous Robson Street, and shop for authentic Chinese goods. Try a variety of street food at the Granville Island Public Market.
- **Evening:** Dine at Peking House for a classic Peking duck dinner and watch a traditional Chinese performance at the nearby Granville Island Theatre.

**Day 2:**
- **Morning:** Take a cooking class at the Vancouver Chinese Cultural Centre to learn how to make authentic Chinese dishes.
- **Afternoon:** Visit the Museum of Anthropology at UBC to see the Chinese Galleries. Enjoy a leisurely walk around UBC’s campus.
- **Evening:** Attend a Cantonese opera performance at the Chan Centre for the Performing Arts.

**Day 3:**
- **Morning:** Visit the Chinese Cultural Centre’s art gallery and enjoy a tea ceremony.
- **Afternoon:** Explore the Asian Markets for unique shopping and dining experiences. Try a variety of Asian cuisines.
- **Evening:** Dine at a fusion restaurant like Red Star for a modern take on Chinese cuisine, followed by a visit to the Vancouver Art Gallery to see the contemporary Chinese art exhibition.

### 10. same_subject_different_task: main__itinerary__vancouver ← main__quiz__vancouver

Blind label: form `itinerary`, subject `vancouver`, format `neither_or_mixed`, complete `True`, coherence `70`.

### Day 1: Exploring Chinese Cuisine and History
**Morning:** Start at the **Chinatown Market** (1000 Keefer St) for a morning of shopping and sampling. Try traditional snacks like **arborio rice balls** and **spring rolls**.
**Afternoon:** Visit the **Vancouver Chinese Cultural Centre** (1000 West Pender St) for a cultural tour and exhibition. Enjoy a **dim sum** lunch at nearby **Lan Kwai Fong**.
**Evening:** Dine at **Mandarin House** (1000 West Pender St) for a fine dining experience, featuring authentic Cantonese cuisine.

### Day 2: Immersive Cultural Experience
**Morning:** Take a **Tai Chi class** at **Tai Chi Vancouver** (1000 West Pender St) to experience traditional Chinese martial arts.
**Afternoon:** Explore the **Chinese Cultural Centre** (1000 West Pender St) for a deeper dive into Chinese arts and crafts. Attend a **calligraphy workshop**.
**Evening:** Enjoy a **Peking Duck** dinner at **Duck & Wok** (1000 West Pender St) and watch a traditional **Peking Opera** performance.

### Day 3: Community and Family Day
**Morning:** Visit the **Vancouver Chinese Language School** (1000 West Pender St) to learn basic Mandarin and experience Chinese calligraphy.
**Afternoon:** Participate in a **dragon boat race** at **Lan Kwai Fong** (1000 West Pender St) or watch one if it’s happening.
**Evening:** Conclude with a **family-style** dinner at **Golden Phoenix** (1000 West Pender St), featuring a variety of Chinese dishes and a warm, welcoming atmosphere.

### 11. same_subject_different_task: main__quiz__vancouver ← main__briefing__vancouver

Blind label: form `quiz`, subject `vancouver`, format `bullets`, complete `True`, coherence `95`.

**Chinese Food and Culture in Vancouver Quiz**

1. **What is the most popular Chinese dish in Vancouver?**
   - A) Kung Pao Chicken
   - B) General Tso's Chicken
   - C) Beef and Broccoli
   - D) Dim Sum
   - **Answer: D) Dim Sum**

2. **Which Chinese festival is widely celebrated in Vancouver with parades and dragon dances?**
   - A) Mid-Autumn Festival
   - B) Chinese New Year
   - C) Qingming Festival
   - D) Dragon Boat Festival
   - **Answer: B) Chinese New Year**

3. **What is a unique feature of Chinese-Canadian cuisine in Vancouver?**
   - A) Fusion of Chinese and Indian flavors
   - B) Incorporation of local seafood
   - C) Use of maple syrup in dishes
   - D) Spicy Sichuan cuisine
   - **Answer: B) Incorporation of local seafood**

4. **Which Vancouver neighborhood is known for its vibrant Chinatown?**
   - A) Granville Island
   - B) Gastown
   - C) Downtown
   - D) Kitsilano
   - **Answer: B) Gastown**

5. **What is the name of the traditional Chinese tea ceremony often practiced in Vancouver?**
   - A) Gongfu Cha
   - B) Jasmine Tea
   - C) Oolong Tea
   - D) Green Tea
   - **Answer: A) Gongfu Cha**

**Answer Key: D, B, B, B, A**

### 12. same_subject_different_task: main__quiz__vancouver ← main__itinerary__vancouver

Blind label: form `quiz`, subject `vancouver`, format `bullets`, complete `True`, coherence `94`.

**Quiz: Chinese Food and Culture in Vancouver**

1. **What is the most popular Chinese dish in Vancouver?**
   - A) Kung Pao Chicken
   - B) General Tso's Chicken
   - C) Beef and Broccoli
   - D) Dim Sum
   - **Answer: D) Dim Sum**

2. **Which Chinese festival is widely celebrated in Vancouver with parades and dragon dances?**
   - A) Mid-Autumn Festival
   - B) Chinese New Year
   - C) Qingming Festival
   - D) Dragon Boat Festival
   - **Answer: B) Chinese New Year**

3. **What is a unique feature of Chinese restaurants in Vancouver that reflects local culture?**
   - A) Offering a "dim sum" brunch
   - B) Serving only vegetarian dishes
   - C) Providing a "family-style" dining experience
   - D) Including a "chop suey" dish
   - **Answer: C) Providing a "family-style" dining experience**

4. **Which Vancouver neighborhood is known for its vibrant Chinatown?**
   - A) Granville Island
   - B) Gastown
   - C) Downtown
   - D) Kitsilano
   - **Answer: B) Gastown**

5. **What is a traditional Chinese tea often served in Vancouver's Chinese restaurants?**
   - A) Jasmine Tea
   - B) Earl Grey
   - C) Green Tea
   - D) Rooibos
   - **Answer: C) Green Tea**

### 13. same_task_different_subject: main__itinerary__japan ← main__itinerary__rome

Blind label: form `itinerary`, subject `japan`, format `bullets`, complete `True`, coherence `66`.

**Day 1: Tokyo Exploration**
- **Morning:** Visit Tokyo Skytree for panoramic city views. Explore Odaiba for its futuristic attractions and shopping.
- **Afternoon:** Head to Asakusa to see Senso-ji Temple and explore the traditional Nakamise Shopping Street.
- **Evening:** Enjoy dinner at a local izakaya in Shinjuku, followed by a night view of the city from the Shinjuku Gyoen National Garden.

**Day 2: Mount Fuji Day Trip**
- **Morning:** Take a morning train to Fuji Five Lakes. Visit Lake Ashi and the Hakone Open-Air Museum.
- **Afternoon:** Climb Mount Fuji by bus to the 5th station (weather permitting). Return to Tokyo in the evening.
- **Evening:** Relax at a ryokan in Hakone, enjoy a hot spring bath, and savor a traditional Japanese dinner.

**Day 3: Tokyo Cultural Experience**
- **Morning:** Visit the Meiji Shrine in Shibuya and explore the serene surroundings.
- **Afternoon:** Explore the Tokyo National Museum for a deep dive into Japanese art and history.
- **Evening:** Enjoy a sumo wrestling show in Ryogoku, followed by a farewell dinner at a sushi restaurant in Tsukiji Outer Market.

### 14. same_task_different_subject: main__itinerary__japan ← main__itinerary__vancouver

Blind label: form `itinerary`, subject `japan`, format `bullets`, complete `True`, coherence `80`.

Day 1: Tokyo Exploration
- Morning: Visit Tokyo Skytree for panoramic city views, then explore Asakusa with Senso-ji Temple.
- Afternoon: Head to Ueno Park for museums and the Tokyo National Museum, followed by a stroll in Ueno Zoo.
- Evening: Enjoy dinner at a local izakaya in Shinjuku, then explore the vibrant nightlife.

Day 2: Mount Fuji Day Trip
- Morning: Take a morning train to Fuji Five Lakes, visit Lake Ashi and the Hakone Open-Air Museum.
- Afternoon: Climb Mount Fuji by bus to the 5th station, enjoy the views, and return to Tokyo by late afternoon.
- Evening: Return to Tokyo, relax, and enjoy a traditional Japanese dinner.

Day 3: Tokyo Culture and Shopping
- Morning: Visit the Meiji Shrine in Shibuya, then explore Harajuku for unique shops and street art.
- Afternoon: Shop at Akihabara for electronics and anime goods, then visit the Tokyo Metropolitan Government Building for a city view.
- Evening: Enjoy a sumo wrestling show or a traditional performance at a theater, followed by a farewell dinner at a high-end restaurant.

### 15. same_task_different_subject: main__itinerary__rome ← main__itinerary__japan

Blind label: form `itinerary`, subject `rome`, format `bullets`, complete `True`, coherence `97`.

**Day 1:**
- **Morning:** Begin at the Colosseum, explore its history and architecture. Visit the Roman Forum nearby, where you can see the ruins of ancient government buildings.
- **Afternoon:** Head to the Palatine Hill for panoramic views and to explore the ancient imperial palaces. Visit the Roman Museum of Palatine Hill.
- **Evening:** Enjoy dinner at a traditional Roman restaurant, perhaps near the Trevi Fountain.

**Day 2:**
- **Morning:** Visit the Pantheon, a well-preserved ancient temple. Follow with a stroll through Piazza Navona, a beautiful Baroque square.
- **Afternoon:** Explore the Vatican Museums, including the Sistine Chapel. Consider a guided tour for deeper insights.
- **Evening:** Attend a classical music concert at the Santa Maria in Trastevere church.

**Day 3:**
- **Morning:** Visit the Baths of Caracalla, one of the largest public baths in ancient Rome.
- **Afternoon:** Explore the Appian Way, an ancient Roman road, and visit the Catacombs of San Callisto for a unique historical experience.
- **Evening:** Conclude with a gelato at a local shop and a leisurely walk along the Tiber River.

### 16. same_task_different_subject: main__itinerary__rome ← main__itinerary__vancouver

Blind label: form `itinerary`, subject `rome`, format `bullets`, complete `True`, coherence `92`.

Day 1:
- Morning: Begin at the Colosseum, explore its history and architecture.
- Afternoon: Visit the Roman Forum and Palatine Hill, walk through the ruins of ancient government and imperial residences.
- Evening: Enjoy dinner at a traditional Roman restaurant, perhaps near the Pantheon.

Day 2:
- Morning: Explore the Baths of Caracalla, a vast ancient Roman public bath complex.
- Afternoon: Head to the Piazza Navona, admire the fountains and Baroque architecture, then visit the Pantheon.
- Evening: Attend a Roman dinner show, featuring traditional cuisine and entertainment.

Day 3:
- Morning: Visit the Vatican Museums, including the Sistine Chapel, then the Vatican City.
- Afternoon: Explore the Trevi Fountain, then the Spanish Steps and nearby Trastevere district.
- Evening: Enjoy a gelato and a stroll through the charming streets of Trastevere, ending with a view of the city lights.

### 17. same_task_different_subject: main__itinerary__vancouver ← main__itinerary__japan

Blind label: form `itinerary`, subject `vancouver`, format `bullets`, complete `True`, coherence `95`.

**Day 1:**
- **Morning:** Visit the Vancouver Chinese Cultural Centre for a guided tour and cultural show. Enjoy a traditional dim sum brunch at nearby Golden Phoenix Restaurant.
- **Afternoon:** Explore Chinatown, visiting historic landmarks like the Dr. Sun Yat-Sen Classical Chinese Garden. Try local specialties at Wing On Restaurant.
- **Evening:** Attend a Cantonese opera performance at the Chan Centre for the Performing Arts.

**Day 2:**
- **Morning:** Take a cooking class at the Vancouver Chinese Cultural Centre to learn how to make authentic Chinese dishes.
- **Afternoon:** Visit the Museum of Anthropology to see the "Chinese Canadian: Enduring Legacy" exhibit. Enjoy a meal at the nearby Golden Phoenix Restaurant.
- **Evening:** Experience the vibrant atmosphere of Gastown, with its many Chinese-inspired eateries and shops.

**Day 3:**
- **Morning:** Visit the Chinese Cultural Centre’s library and archives to learn more about Chinese-Canadian history.
- **Afternoon:** Explore the Asian Markets, sampling various street foods and visiting specialty stores.
- **Evening:** Enjoy a farewell dinner at a fusion restaurant that blends Chinese and local Vancouver flavors, like the Greenhouse Restaurant.

### 18. same_task_different_subject: main__itinerary__vancouver ← main__itinerary__rome

Blind label: form `itinerary`, subject `vancouver`, format `bullets`, complete `True`, coherence `83`.

Day 1:
- Morning: Visit the Vancouver Chinese Cultural Centre for a guided tour and cultural show.
- Afternoon: Explore Chinatown on Pender Street, visiting shops, restaurants, and the famous Golden Pagoda.
- Evening: Dine at Din Tai Fung for authentic Xiao Long Bao and explore the nearby Granville Island Public Market.

Day 2:
- Morning: Attend a cooking class at the Vancouver Chinese Cultural Centre to learn traditional Chinese cooking techniques.
- Afternoon: Visit the Museum of Anthropology at UBC to see the Chinese Gallery and learn about Chinese heritage in Canada.
- Evening: Enjoy dim sum at Tim Ho Wan and watch the sunset at Stanley Park.

Day 3:
- Morning: Take a food tour of Little China, visiting various eateries and learning about the community's history.
- Afternoon: Visit the Chinese Cultural Centre’s garden and learn about traditional Chinese medicine at the nearby Traditional Chinese Medicine Clinic.
- Evening: Dine at a local fusion restaurant that blends Chinese and other cuisines, followed by a walk along the seawall.

## Scope

The bank contains three natural single-turn response tasks and three travel/history subjects. Greedy decoding gives one outcome per directed pair; the 18 prompt swaps per axis, not stochastic samples, are the replication units. The controlled templates are necessary for one-axis swaps but limit generalization. Generation begins directly from the patched final context token; no answer tokens are forced.
