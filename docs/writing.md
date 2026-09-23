# Writing rules for this repository

We write code comments and documents in Simplified Technical English. The rules
come from ASD-STE100 and from the SimpleEnglish skill.

The reader is tired. The reader must not misread a sentence.

## The rules

| Rule | What it removes |
|---|---|
| Max 20 words per instruction. Max 25 words per description. | The run-on sentence |
| Condition before command. | A reader who acts too early |
| Simple tenses. Active voice. | "has been updated" |
| No should, would, may, or might. Use can, will, or must. | Hedging |
| One word has one meaning in the whole document. | check/verify/confirm roulette |
| Keep the articles. Keep "that". | Telegraph style |
| No em-dash. | The half thought |
| No bold lead-in. No heading over two sentences. | Decoration |
| State the fact. Do not state its importance. | "crucial", "robust" |

## How we use the rules here

* The code stays as it is. Only the comments and the documents change.
* Identifiers and metric names stay in their current form.
* We write English in new files and in files that we touch.
* Older files keep their language until a later pass.
* We do not use the STE dictionary. We use the plain register.

## One example

Before:

> The loader checks the header of the npy file, and this matters a lot because
> a C-order array silently transposes the cache — which cost us 6 arms.

After:

> The loader reads the header of the npy file. A C-order array transposes the
> cache and the runs fail later. This fault cost 6 arms.
