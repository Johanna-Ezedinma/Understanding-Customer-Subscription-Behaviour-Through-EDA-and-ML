# EDA Summary

[![README](https://img.shields.io/badge/README-181717?style=for-the-badge&logo=git&logoColor=white)](../README.md)

## Overall Conversion

47.4% of the 11,162 customers contacted subscribed to a term deposit (5,289 of 11,162).

## Campaign Performance & ROI

**Past campaign outcome is the single strongest signal found in this analysis.** Customers whose previous
campaign outcome was a success converted at **91.3%** this time, versus 50.3% for a past failure, 57.2% for
"other," and 40.7% for customers with no previous campaign to report (`unknown`).

**More contact attempts do not help, they hurt.** Conversion drops steadily as contact attempts increase:
53% on the first attempt, down to 46% (2-3 attempts), 39% (4-6), and just 27% at 7+ attempts. There is a
clear point of diminishing returns, and heavy recontacting appears actively counterproductive.

## Customer Targeting & Segmentation

**Job:** Students (75%) and retirees (66%) convert far above average. Blue-collar workers (36%) and
entrepreneurs (38%) convert well below average.

**Age:** Conversion is U-shaped, not linear. The youngest (18-25: 71%) and oldest (66+: 80%) age groups
convert far better than the middle of the working-age range (36-45: 41%, 46-55: 42%), which converts worst
of all. Combined with the job finding above, this points toward students and retirees, not typical
working-age customers, as the strongest segment.

**Education:** Tertiary-educated customers convert best (54%), primary the worst (39%). The gap is real but
smaller than the job or age effect.

**Marital status:** Single customers convert best (54%), married customers worst (43%). A modest effect.

**Balance:** Customers with a negative (overdrawn) balance convert dramatically worse (34%) than any other
tier, including the "Low" tier (45%), "Medium" (55%), and "High" (57%). Overdrawn customers may be a segment
worth deprioritizing or approaching differently.

## Timing & Outreach Strategy

**Month shows the widest swing of any single factor in this dataset.** December (91%), March (90%), and
September (84%) are the strongest months by a wide margin, while May (33%), January (41%), and July (41%)
are the weakest. This is a far larger spread than any customer-level segment above, timing may matter more
than who is being called.

**Day of month** shows no clean pattern; conversion bounces between roughly 32% and 74% without an obvious
trend across the month.

**Contact method matters substantially.** Cellular (54%) and telephone (50%) convert similarly, but
`unknown` contact method converts far worse (23%), less than half the rate of a known channel.

## Financial Risk & Product Fit

**Housing and personal loans both independently reduce conversion, and the effect stacks.** Customers with
neither loan convert best (60%). Having a housing loan alone drops this to 38%; a personal loan alone drops
it to 35%; having both drops it to 32%, the worst-converting group in this cross-tab.

**Credit in default customers still convert, but at a reduced rate** (31% vs. 48% for those without),
suggesting exclusion isn't clearly warranted, but expectations should be adjusted downward for this group.

## Operational & Process

**Prior contact makes a large difference.** Customers previously contacted convert at 67%, compared to 41%
for customers reached cold, a 26-point gap, reinforcing the `poutcome_success` finding above from a
different angle.

**Call duration correlates strongly with outcome**, successful calls average 537 seconds (median 426),
versus 223 seconds (median 163) for unsuccessful calls. This is a genuinely strong descriptive pattern, but
`duration` is only known _after_ a call happens, so this is useful for agent coaching insight, not for
deciding who to call in advance (see the machine learning summary for how this was handled in modeling).

## Summary of Strongest Patterns (ranked)

1. **Past campaign success** (91% vs. 41-57% otherwise) — the strongest single predictor found in EDA.
2. **Month of contact** (91% in December vs. 33% in May) — the widest swing of any factor.
3. **Job and age together** point to students and retirees as the standout segment.
4. **Existing loans** (housing and/or personal) meaningfully suppress conversion, and the effect compounds.
5. **Contact method being unknown** roughly halves conversion versus a known channel.
