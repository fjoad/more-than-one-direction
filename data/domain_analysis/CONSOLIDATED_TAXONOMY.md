# Consolidated data-derived domain taxonomy

## How this taxonomy was obtained

The first pass assigned one unrestricted, open-ended subject label to each of the 2,054 prompts. It produced 1,119 distinct phrasings. Three independent reviews then grouped those observed labels into approximately 12–18 coherent subject areas. Their proposals converged on the taxonomy below. No split labels, source-dataset categories, refusal types, or safety categories were used to define it.

The final taxonomy contains **17 substantive domains plus one administrative fallback**. Every prompt receives exactly one label. Assignment follows the prompt's primary literal subject, not its harmfulness, requested format, truth value, or likely reason for refusal.

## Final domains

1. **Technology, AI & Cybersecurity** — Computing, software, devices, engineering technology, artificial intelligence, networks, hacking, malware, phishing, and system/account security. Requests whose central object is a person's data go to Privacy.
2. **Privacy & Personal Data** — Obtaining, exposing, tracking, inferring, or protecting identifiable contact, location, health, financial, biometric, employment, identity, account, or communications data.
3. **Finance, Business & Economics** — Banking, investing, markets, cryptocurrency, taxation, insurance, commerce, entrepreneurship, corporations, management, and economics. Operational fraud goes to Crime.
4. **Crime, Fraud & Illicit Activity** — Theft, burglary, robbery when property-taking is primary, scams, forgery, counterfeiting, smuggling, trafficking, corruption, money laundering, vandalism, concealment, and non-cyber illicit schemes.
5. **Violence, Weapons & Physical Safety** — Physical harm, homicide, assault, abuse, kidnapping, terrorism centered on attack, weapons, explosives, harmful poisoning, arson, and mass or public-safety threats.
6. **Health, Medicine & Mental Health** — Diagnosis, treatment, anatomy, public health, reproductive health, medical research, mental illness, self-harm, suicide, and health-centered nutrition. Recreational substance use goes to Drugs.
7. **Drugs, Alcohol & Addiction** — Illegal or recreational drugs, alcohol, tobacco, intoxication, overdose, addiction, manufacture, cultivation, trafficking, prescription misuse, doping, and recovery.
8. **Sexuality & Adult Content** — Consensual sexual behavior, sexual health when sex is primary, erotic material, pornography, sexual roleplay, fetishes, sex work, and adult entertainment. Sexual coercion and child sexual exploitation go to Violence.
9. **Relationships, Family & Personal Life** — Dating, marriage, friendship, parenting, family structure, interpersonal communication and conflict, emotional manipulation, reputation, personal futures, and ordinary social conduct.
10. **Identity, Discrimination & Social Justice** — Race, ethnicity, nationality, gender, LGBTQ identity, disability, age, class, stereotypes, prejudice, exclusion, equality, civil rights, and group-based oppression.
11. **Politics, Government, Law & International Affairs** — Elections, political actors and ideologies, courts, legislation, public policy, government administration, diplomacy, intelligence, geopolitics, and current state affairs.
12. **Education, Employment & Workplace** — Schools, universities, teaching, admissions, credentials, careers, hiring, labor, wages, workplace conduct, and organizational practices.
13. **Arts, Media, Literature & Language** — Books, film, television, music, visual and performing arts, games as media, publishing, journalism as media, creative works, language, translation, and transcription.
14. **Religion, Philosophy & Ethics** — Religious traditions and practices, theology, spirituality, atheism, mythology as belief, philosophy, and explicitly normative ethical questions.
15. **Science, Nature & Environment** — Mathematics, physics, chemistry, astronomy, space, non-clinical biology, genetics, earth science, climate, ecology, geography, animals, plants, agriculture, pollution, conservation, and natural disasters.
16. **History, Geography & Cultural Heritage** — Past events, people and institutions, historical wars and atrocities, archaeology, museums, heritage, historical places, and cultural traditions when their historical context is primary. Current affairs go to Politics.
17. **Lifestyle, Food, Sports & Recreation** — Cooking, cuisine, travel, tourism, fashion, home life, consumer routines, sports, fitness, hobbies, games as play, outdoor recreation, and other ordinary practical topics.
18. **No Discernible Subject** — Strict fallback for word salad, unresolved placeholders, or fragments from which no stable literal topic can be recovered. Vagueness, impossibility, or a false premise alone does not justify this label.

## Tie-breaking rules

- Choose the subject that occupies most of the prompt or is necessary to answer it.
- Classify misinformation and conspiracy claims by their underlying subject.
- Classify a request for a person's private datum as Privacy even when the datum is financial, medical, educational, or political.
- Use Technology when system intrusion is the central operation; use Privacy when personal information is the central object.
- Use Violence for physical injury, sexual coercion, weapons, explosives, and harmful poisoning; use Crime for primarily nonviolent illicit acquisition, fraud, concealment, or trade.
- Use Identity when unequal treatment of a social group is central, even if the setting is a school, workplace, religion, or political debate.
- Use History for retrospective events; use Politics for contemporary governance, advocacy, law, diplomacy, or public policy.
- Use No Discernible Subject only after every substantive domain has been ruled out.
