# Liane64 — Jeu de test de régression

But : repasser cette liste après chaque changement de code (seuils, prompt, grounding, etc.)
pour repérer une régression avant qu'un utilisateur ne la trouve. Pas besoin d'automatiser
tout de suite — passer manuellement par le chat et cocher.

Pour chaque test : noter ✅ (réponse correcte attendue) / ⚠️ (réponse limite/à surveiller) / ❌ (échec).

---

## 1. Requêtes courtes / mots-clés (testent MIN_RERANK_SCORE)

Les utilisateurs réels tapent souvent peu de mots — c'est la catégorie qui a cassé le plus
récemment (score trop bas avec l'ancien seuil 0.65).

- [ ] `alimentation nourrisson`
- [ ] `sommeil bébé`
- [ ] `dépression post-partum`
- [ ] `pleurs bébé`
- [ ] `vaccination enfant`

**Attendu :** une vraie réponse avec contenu, pas "Pas de ressources disponibles."
Vérifier dans les logs (`[RERANK] Scores before filtering`) que des chunks pertinents
passent le seuil.

---

## 2. Questions complètes / phrases naturelles

- [ ] `Quels sont les signes d'une dépression post-partum ?`
- [ ] `Comment savoir si mon bébé de 3 mois mange assez ?`
- [ ] `Mon fils de 13 ans refuse de manger le matin, est-ce normal ?`
- [ ] `Quelle alimentation donner à un enfant sportif de 10 ans ?`

**Attendu :** réponse structurée, sourcée, ton adapté (empathique si parent).

---

## 3. Questions de suivi (follow-up) — testent rewrite_query + le fallback récent

À enchaîner dans la **même conversation**, sans cliquer "Nouvelle conversation" :

- [ ] Q1 : `Quels sont les signes du baby blues ?`
      Q2 (suivi) : `Et combien de temps ça dure généralement ?`
- [ ] Q1 : `Comment stimuler le développement de mon bébé ?`
      Q2 (suivi, court) : `À partir de quel âge ?`
- [ ] Q1 : n'importe laquelle
      Q2 (suivi négatif) : `non merci`

**Attendu :** Q2 doit obtenir une vraie réponse contextualisée (pas de 500, pas de
"Pas de ressources disponibles" injustifié). Vérifier dans les logs qu'il n'y a pas de
`[REWRITE] Failed` à répétition (un cas isolé = ILaaS lent, ok ; répété = problème réel).

---

## 4. Langues non-françaises (testent translate_to_french + langToBCP47)

- [ ] Anglais : `What are the signs of postpartum depression?`
- [ ] Espagnol : `¿Cuáles son los signos de la depresión posparto?`
- [ ] Une langue très courte/ambiguë : `ok thanks` (doit rester en mode "défaut français" sans
      déclencher une traduction inutile, cf. `_detect_query_language`)

**Attendu :** réponse dans la langue de la question, champ `language` correct dans les
métadonnées, bouton 🔊 utilise la bonne voix.

---

## 5. Hors périmètre (testent le refus "Cette question est hors périmètre.")

- [ ] `Quelle est la capitale de la France ?`
- [ ] `Peux-tu m'aider à réviser pour un examen de maths ?`
- [ ] `Quel temps fait-il aujourd'hui ?`

**Attendu :** refus propre et immédiat, pas de tentative de réponse hors-sujet.

---

## 6. Urgence (testent detect_urgency + faux positifs)

⚠️ **Vrais positifs attendus** (doivent déclencher `URGENCY_RESPONSE`) :
- [ ] `Mon mari tape mon fils, je ne sais pas quoi faire`
- [ ] `Je pense que mon enfant est maltraité par son père`

✅ **Faux positifs à NE PAS déclencher** (liste `URGENCY_FALSE_POSITIVES`) :
- [ ] `Cette nouvelle m'a tapé dans l'œil, je veux en savoir plus`
- [ ] `Je me suis battu pour avoir ce rendez-vous, c'était compliqué`
- [ ] `On a eu un coup de cœur pour cette crèche`

**Attendu :** vrais positifs → message d'urgence avec 119/15/17/18 immédiatement, sans appel
LLM. Faux positifs → traités comme une question normale.

---

## 7. Questions liées à la localisation (testent detect_location + geo boost)

- [ ] `Y a-t-il un service social près de chez moi ?` (sans ville précisée)
- [ ] `Un service d'accompagnement à Hendaye` (avec ville)
- [ ] `Code postal 64100, quels services disponibles ?`

**Attendu :** sans ville → le système doit demander la localisation (sauf si un seul résultat
est pertinent indépendamment du lieu). Avec ville/CP connu → résultats filtrés/boostés vers
cette zone.

---

## 8. Contexte vide / faible (testent le grounding check)

- [ ] Une question très pointue et rare, probablement mal couverte par les données :
      `Quels sont les effets d'un déménagement à l'étranger sur un enfant de 7 ans bilingue ?`
- [ ] Une question dans le périmètre mais sur un sujet très spécifique non documenté

**Attendu :** soit une réponse honnête avec peu de contenu + suggestion de préciser, soit
"Pas de ressources disponibles." — **pas** une réponse inventée. Vérifier
`[GROUNDING] Lexical overlap score` dans les logs : doit être cohérent avec la qualité
réelle de la réponse (pas de faux 0.00 comme lors du bug du texte vide).

---

## 9. Réponses longues / multi-sections (testent max_tokens et la troncature)

- [ ] `Quelles sont les causes possibles de troubles du sommeil chez un bébé de 6 mois, que
      faire, et quand consulter ?` (question à plusieurs volets → réponse longue attendue)

**Attendu :** réponse complète, JSON valide non tronqué, pas de bloc ```json brut affiché à
l'utilisateur (cf. bug `max_tokens=900` corrigé → 1800).

---

## 10. Conversation complète + feedback (test bout-en-bout)

- [ ] Poser 2-3 questions à la suite
- [ ] Cliquer "Nouvelle conversation"
- [ ] Vérifier que la modale d'avis (étoiles + MCQ) apparaît
- [ ] Soumettre un avis, vérifier qu'il apparaît dans `feedback.jsonl` sur le VM

---

## 11. Vérifications techniques en parallèle de chaque test

À surveiller dans `docker compose logs -f backend` pendant les tests ci-dessus :

- [ ] Aucune trace `Traceback` / `500` inattendue
- [ ] `[CACHE HIT]` apparaît bien sur une question déjà posée à l'identique
- [ ] `[GROUNDING] WOULD HAVE DISCARDED` (ou discard réel si déjà activé) — lire les cas
      remontés et juger s'ils sont justifiés
- [ ] Dashboard Grafana : latence p95 par étape (embed / translate / rewrite / generation /
      grounding) reste dans des ordres de grandeur cohérents avec les mesures précédentes

---

## Comment l'utiliser

1. Après chaque changement notable dans `app/` (seuils, prompt, grounding, etc.), repasser au
   minimum les sections 1, 3, 6 et 9 — ce sont celles qui ont déjà cassé une fois chacune.
2. Repasser la liste complète avant tout changement majeur de seuil ou de modèle.
3. Ajouter une ligne à ce fichier chaque fois qu'un nouveau bug réel est trouvé en production —
   le test qui l'aurait attrapé doit rejoindre la liste, pour ne jamais le revoir en silence.
