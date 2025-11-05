**Study Objectives**
The objective of this study is to systematically extract and curate data related to HIV drug resistance. Specifically, the study aims to identify and compile information from published reports that include:

1. HIV sequences obtained from patient samples, including their corresponding GenBank accession numbers.
2. The demographics of the populations undergoing HIV sequencing
3. Details on the samples and sequencing methods used to generate these sequences.
4. Antiretroviral treatment histories of the persons undergoing HIV sequencing.

**Question 1: Does the paper report HIV sequences from patient samples?**

1. The answer is likely to be “Yes” if :
    1.1. The paper states sequences from clinical samples such as “plasma”, “serum”, “PBMC”, “buffy coat”, or “dried blood spots \(DBS\)”.
    1.2. The paper states that sequenced samples were obtained from “patients”, “participants”, “subjects”, “blood donors”, “newly diagnosed persons”, “newly infected persons”, “PWH”, “PLWH” or “cohorts”.
    1.3. The paper refers to “genotypic resistance testing”, “HIV sequencing”, “Sanger”, “Illumina” “MiSeq”, “single-genome”, or “whole genome” sequencing.
    1.4. The paper reports “GenBank accession numbers” for sequences generated in this study or states sequences were “submitted/deposited”.
    1.5. The paper mentions sequencing and “ART-naïve” or “ART-experienced” patients, “baseline” or “virologic failure \(VF\)” samples, “transmitted drug resistance \(TDR\)”, “HIV drug resistance surveillance”.

2. The answer is likely to be “No” if :
    2.1. Only laboratory strains or site-directed mutants were reported or sequenced. Examples of laboratory strains include HXB2, IIIB, NL43, BAL, and LAI.
    2.2. Work in the paper is limited to just susceptibility testing or biochemical enzyme or replication fitness studies on laboratory strains or on panels of mutant strains.
    2.3. The paper is a review or meta-analysis that solely analyzes sequences downloaded from databases or reported in other studies.
    2.4. Only clinical or virological outcomes or pharmacokinetics are reported.

**Question 2: Does the paper report in vitro drug susceptibility data?**

1. The answer is likely to be “Yes” if :
    1.1. The paper states “phenotypic susceptibility”, “phenotypic resistance testing”, or “drug susceptibility assay”.
    1.2. The methods or results report IC50, EC50, EC90, fold-change in susceptibility, or reduced susceptibility.
    1.3. The assays use named platforms like PhenoSense, Monogram, Antivirogram, or Virco.
    1.4. Cell-based susceptibility assays are described \(e.g., TZM-bl, MAGIC-5, MT-2, PBMCs, SupT1, CEM-ss, HeLa-CD4, HEK293\) with drug titrations and reporter readouts \(luciferase, p24, RT activity\).
    1.5. The study uses recombinant/pseudotyped/site-directed mutant viruses or patient-derived clones and tests antiviral activity in vitro against drugs.
    1.6. In vitro enzymatic inhibition assays or resistance passage/selection experiments with subsequent phenotypic susceptibility measures are reported.

2. The answer is likely to be “No” if :
    2.1. Only genotypic resistance, mutation prevalence, or algorithmic/predicted susceptibility \(Stanford HIVdb, ANRS, Rega, Geno2pheno, WHO SDRM/CPR\) is reported without reporting in vitro susceptibility data.
    2.2. The study focuses on sequencing, phylogenetics, recombination, epidemiology, surveillance, prevalence of transmitted drug resistance \(TDR\), or transmission clusters without reporting in vitro susceptibility data.
    2.3. Only replication capacity/fitness assays are reported without drugs.
    2.4. Only tropism phenotyping \(e.g., Trofile R5/X4\) is reported.
    2.5. Only clinical outcomes, pharmacokinetics/drug levels, or case descriptions without in vitro susceptibility data are reported.
    2.6. In vitro selection/passaging is described but no susceptibility data are reported.

**Question 3: Were sequences from the paper made publicly available?**
**If the answer to the first question – Does the paper report HIV sequences from patient samples? – is “No”, then the answer is “No”. Otherwise, read the remainder of the “if-then” rules for this question.**

1. The answer is likely to be “Yes” if :
    1.1. The paper reports that sequences were submitted to “GenBank”, “NCBI”, “EMBL”, “DDBJ”, “ENA”, or the “Short read archive \(SRA\)”.
    1.2. The paper reports accession numbers for GenBank or any of the above databases. Accession numbers contain two letters followed by a series of numbers.
    1.3. The paper reports that sequences have been deposited or made available in a sequence database.
    1.4. The paper reports that the raw reads are in the SRA database or consensus/amplicon sequences are in GenBank with accession numbers.
    1.5. Accession numbers are provided in a Supplementary Table or in the Data Availability statement.
    1.6. The authors state that sequences were submitted to GenBank but that the accession numbers are pending.

2. The answer is likely to be “No” if :
    2.1. The only accession numbers reported are for HIV-1 reference strains such as HXB2 \(K03455.1\), NL4-3 \(AF324493.2\), LAI \(U54771\), and BAL \(AY713409\).
    2.2. No accession numbers are reported and no explicit deposition statement to public repositories is made.
    2.3. The paper only cites accession numbers from previous studies or reference strains.
    2.4. The paper is a review or meta-analysis using existing just existing sequence data.

**Question 4: What were the GenBank accession numbers for sequenced HIV isolates?**
**If the answer to questions 1 or 3 is “No”, then report “Not reported”. Otherwise, read the remainder of the “if-then” rules for this question.**

1. If the paper reports that sequences have been submitted or deposited to GenBank and if accession numbers have been provided in the text then extract the ranges and individual accession numbers \(e.g., “FJ800379–FJ800386”, “GQ477441-GQ477451”, “KP170487”\).
2. The accession numbers could be presented in the Methods, in a Table, or in a Data Availability statement then extract the ranges and individual accessions.
3. If you are ensure of how to extract the accession numbers \(e.g., ranges and individual numbers\) then report the text containing these numbers exactly as printed.
4. If the paper reports accession numbers for sequences but refers to other public sequence databases, then extract the accession numbers because the nucleotide sequence databases share their sequences under the same accession numbers.

**Question 5: How many individuals had samples obtained for HIV sequencing?**
**If the answer to question 1 is “No”, then the answer to this question is “0”. Otherwise, read the remainder of the “if-then” rules for this question.**

1. If the paper reports that “genotypic resistance testing”, “genotyping”, or “sequencing” was performed on samples from X number of individuals, then report the number X.
2. If the paper reports that X sequences were reported from Y individuals, then report the number Y.
3. If the paper reports that a panel of X isolates from the plasma of Y treatment-naïve or experienced individuals, then report the total number of individuals, which is Y.
4. If the paper reports that X isolates were sequenced from a group of patients, then unless stated otherwise assume that one sequence was obtained per individual and report the number X.
5. If the paper reports the numbers of sequences for each of different mutually exclusive groups of patients, then report the total number of patients from whom sequences were obtained.
6. If the paper is a case report then report 1. If it is a case series, then report the number of persons in the case series.
7. If the paper reports sequences are compiled solely from a database or previously published studies then report “0”.
8. If the papers reports that sequences were obtained on patients in the study but the number of individuals or samples undergoing sequencing is not reported, then report “Not reported”.

**Question 6: From which countries were the sequenced samples obtained?**
**If the answer to question 1 is “No”, then the answer to this question is “Not applicable”. Otherwise, read the remainder of the “if-then” rules for this question.**

1. If the paper states patients or participants were recruited in, enrolled in, or followed at hospital or clinic in &lt;country\&gt;, then use that country as the sample origin.
2. If the paper states that samples were collected in &lt;country\&gt;, then use that country as the sample origin.
3. If the paper mentions nationwide “cohort” or “surveillance” or “registry” in &lt;country\&gt; or “national reference laboratory” of &lt;country\&gt;, then use that country as the sample origin.
4. If the paper names a province or city uniquely tied to a country, report that country.
5. If the paper explicitly lists multicenter/enrollment sites across countries e.g., “enrolled at sites in Botswana, Uganda, USA, Zimbabwe”, then list all named countries: Botswana, Uganda, USA, Zimbabwe.
6. If the paper mentions “study conducted in &lt;country 1\&gt;, samples shipped to &lt;country 2\&gt; for sequencing”, then use the collection country &lt;country 1\&gt; as the sample origin.
7. If the paper references a country-specific cohort name \(e.g., SCOPE—San Francisco; ATHENA—Netherlands; SHCS—Switzerland; ANRS CO5 HIV-2—France; BHP—Botswana; UARTO—Uganda\), then use the cohort’s country as the sample origin.
8. If participants are immigrants but samples were drawn in the host country, then use the host country as the sample origin.
9. If the paper does not explicitly provide names of countries or nationalities, answer “Not Reported”.

**Question 7: From what years were the sequenced samples obtained?**
**If the answer to question 1 is “No”, then the answer to this question is “Not applicable”. Otherwise, read the remainder of the “if-then” rules for this question.**

1. If the paper reports that clinical samples were collected between &lt;Year1\&gt; to &lt;Year2\&gt; or &lt;Month/Year1\&gt; to &lt;Month/Year2\&gt; then extract that year range \(e.g., “between June 2004 and April 2007”, then report 2004–2007\).
2. If the paper lists per-year counts \(“collected in 2008, 2009, 2010…”\) then report the full span \(e.g., 2008–2017\).
3. If the paper reports months only \(e.g., “March–September 2008”\), then report the single year \(2008\).
4. If the paper states patients were enrolled between &lt;year1\&gt;-&lt;year2\&gt; and baseline genotyping/sequencing was done at enrollment, then use those enrollment years.
5. If the paper states patients were diagnosed between &lt;year1\&gt;-&lt;year2\&gt; and “baseline sequences” or “all baseline sequences” were included, then use the diagnosis range.

Answer the question as “Not reported” if:

1. The paper does not report calendar years. Do not infer years
2. Sequencing was done but years are not reported.
3. The paper reports publication years or trial end dates not explicitly tied to sample collection.

**Question 8: Were samples cloned prior to sequencing?**
**If the answer to question 1 is “No”, then the answer to this question is “Not applicable”. Otherwise, read the remainder of the “if-then” rules for this question.**

1. The answer is likely to be “Yes” if :
    1.1. The paper states “amplicons were cloned and sequenced”, “all products were cloned before Sanger sequencing”, “TA cloning/Topo TA cloning was performed”, or cites a cloning kit \(e.g., NEB PCR cloning kit\).
    1.2. The paper describes “clonal analyses” or “clonal sequencing” of patient-derived amplicons or plasma, such as clone-level integrase genotypes.
    1.3. The paper states “molecular clones” or “PCR product was cloned into a vector and then sequenced”.
    1.4. The paper describes use of “single-genome amplification \(SGA\)”, “single-proviral sequencing”, or “limiting dilution for direct sequencing”.

2. The answer is likely to be “No” if :
    2.1. The paper includes phrases such as “direct sequencing”, “sequenced directly”, “population-based”, “bulk/consensus sequencing”, or “both strands directly sequenced”.
    2.2. The paper describes use of certain clinical genotyping platforms that do not require cloning e.g. ViroSeq, TruGene, GeneSeqR, GenoSure, Abbott/ABI BigDye/3130/3500/3730 analyzers.
    2.3. The paper describes use of “RT-PCR followed by Sanger” or “direct sequencing of PCR amplicons”.
    2.4. The paper describes use of NGS on amplicons or libraries without cloning terms, e.g. Illumina MiSeq/HiSeq/454/GS Junior; “captured libraries” or “molecular barcodes/UMIs”.

3. Answer the question as “Not Reported” if:
    3.1. The methods and results sections lack information about how sequencing was performed.
    3.2. The study reports resistance or clinical outcomes without describing sequencing workflow.

**Question 9: Which HIV genes were reported to have been sequenced?**
**If the answer to question 1 is “No”, then the answer to this question is “Not applicable”. Otherwise, read the remainder of the “if-then” rules for this question.**

1. If the paper states “pol genotyping/sequencing” without providing individual genes, then report “Pol”.
2. If the paper states “pol region encompassing protease and reverse transcriptase”, “PR/RT”, “ViroSeq”, or “TruGene \(PR/RT\)” then answer is likely to have “PR, RT”.
3. If the paper states “integrase coding region of pol”, “IN”, “p31”, “GeneSeq/ViroSeq Integrase”, “INSTI resistance”, “integrase sequencing”, or “GeneSeq/PhenoSense Integrase”, then answer is likely to have “IN”.
4. If the paper states “reverse transcriptase \(RT\)”, “p51/p66”, “NRTI or NNRTI resistance” or “reverse transcriptase sequencing”,  then answer is likely to have “RT”.
5. If the paper states “protease \(PR\)” with codons ~1–99 or PR amplicon, “PI resistance”, or “protease sequencing”, then answer is likely to have “PR”.
6. If the paper mentions lenacapavir, then answer is likely to have “CA”.
7. If the paper mentions “env”, “gp120”, “gp41”, “gp160”, “V3 loop/C2V3”, or “coreceptor usage geno2pheno”, then answer is likely to have “Env”.
8. If the paper mentions “gag”, “p17”, “p24”, “capsid/CA”, “p6”, then consider Gag; if “Gag-protease”/“Gag-PR”, then answer is likely to have “Gag, PR”.
9. If the paper mentions “p6-RT”/“p6PrRT”, then answer is likely to have “Gag, PR, RT”.
10. If the paper states that the sequencing was done on the “full-length genome”, “whole genome”, “complete genome”, or “WGS”, then report “Full length genome” as the answer.
11. If the paper states that sequencing was done on “near full-length genome”, “NFLG”, or “5′/3′ half-genome”, then report “Near full length genome” as the answer.

**Question 10: What method was used for sequencing?**
**If the answer to question 1 is “No”, then the answer to this question is “Not applicable”. Otherwise, read the remainder of the “if-then” rules for this question.**

1. The answer is likely to be “Sanger sequencing” if :
    1.1. The paper mentions ABI instruments \(ABI PRISM/Applied Biosystems 310/3100/3130/3130xl/3500/3500xl/3730/3730xl/3100-Avant\) or Beckman CEQ 2000 XL.
    1.2. The paper mentions BigDye/ABI PRISM Dye Terminator \(v1.0/v3.0/v3.1\), dye-terminator, dye–deoxy chain terminator, or dideoxy sequencing.
    1.3. The paper states capillary sequencer/capillary electrophoresis with ABI analyzers, or electropherograms.
    1.4. The paper states “direct PCR sequencing”.
    1.5. The paper was published before 2007.
    1.6. The paper uses clinical genotyping kits TruGene/OpenGene/ViroSeq/GeneSeq Integrase/vircoTYPE/ANRS consensus technique on ABI platforms.
    1.7. The paper references Sequencher/SeqScape as analysis tools alongside ABI capillary data.

2. The answer is likely to be “NGS” if :
    2.1. The paper mentions Illumina platforms \(MiSeq/HiSeq/iSeq\), paired-end, library prep \(Nextera, NEBNext\), target capture, or “Illumina platform”
    2.2. The paper states “next-generation sequencing \(NGS\)”, “ultra-deep sequencing \(UDS\)”, or “deep sequencing” without a platform.
    2.3. The paper specifies 454/pyrosequencing \(GS Junior/GS Titanium\), GenoSure Gag-Pro “uses NGS”, Sentosa SQ/Vela NGS, or Oxford Nanopore \(GridION\).
    2.4. The paper specifies PacBio RSII/SMRT as the sequencing method

3. Answer the question as “Not Reported” if the paper does not describe the sequencing approach.

**Question 11: What type of samples were sequenced?**
**If the answer to question 1 is “No”, then the answer to this question is “Not applicable”. Otherwise, read the remainder of the “if-then” rules for this question.**

1. The answer is likely to contain “Plasma” if:
    1.1. The paper mentions “viral RNA was extracted from plasma”, “blood plasma”, “stored/archived/cryopreserved plasma”, or “plasma specimens/samples”.
    1.2. The methods mention “plasma HIV-1 RNA” with RT-PCR/sequencing/genotyping.
    1.3. The paper specifies sequencing at “baseline”, “screening”, “virologic failure” or “rebound” uses plasma VL thresholds \(≥400–1000 copies/mL\) or “sequences amplified from patient plasma”.
    1.4. The paper describes “plasma-derived” sequences or “recombinant viruses encoding plasma-derived” gene regions.
    1.5. The paper states that “serum” \(e.g., “blood serum”\) was used for RNA extraction/genome amplification.

2. The answer is likely to contain “PBMC” if:
    2.1. The paper states “PBMC”, “peripheral blood mononuclear cells”, “proviral DNA”, “cell-associated DNA/RNA”.
    2.2. The paper mentions “Archived HIV”.
    2.3. The paper states “whole-virus isolation from PBMCs”, “NFL proviral sequencing”, or “single-proviral sequencing from PBMC DNA”.

3. The answer is likely to contain “Whole Blood” if:
    3.1. The paper states that DNA was extracted from “whole blood” with sequencing/genotyping.
    3.2. The paper states that nucleic acids were extracted from “plasma or whole blood” and DNA from whole blood was sequenced.
    3.3. The source for sequencing were “buffy coat” samples.

4. The answer is likely to contain “DBS” if “dried blood spots” or “DBS” were used for genotyping/sequencing.

5. The answer is likely to contain Lymph node if “LNMCs/lymph node mononuclear cells” or “gut tissue” was sequenced.

6. If the paper explicitly sequences from more than one sample types e.g., Plasma and PBMC; Plasma and DBS, report all mentioned types.

7. If viral sequence results were reported without indicating the sequencing source, answer “Not Reported”.

**Question 12: Were any sequences obtained from individuals with virological failure on a treatment regimen?**
**If the answer to question 1 is “No”, then the answer to this question is “Not applicable”. Otherwise, read the remainder of the “if-then” rules for this question.**

1. The answer is likely to be “Yes” if :
    1.1. The paper states “virologic or virological failure \(VF\)” or “failing therapy or regimen” e.g., failing NNRTI/PI/INSTI or “salvage regimen”.
    1.2. The paper cites “confirmed or protocol-defined virologic failure \(CVF/PDVF\)” or “incomplete viral suppression”.
    1.3. The paper reports that sequencing or genotyping were done for participants with VL \&gt;50 copies/mL while on ART.
    1.4. The paper uses terms like “not virally suppressed on ART”, “viremic on ART”, “viral rebound while on therapy”, or “unsuppressed while treated”.
    1.5. The paper reports genotypic resistance testing or sequencing at the time of “VF” or “failure”, with “history of VF”, or meeting “VF criteria”.
    1.6. The paper suggests that “treatment-emergent resistance”, “emergence of resistance mutations during regimen”, or “switch due to failure” are linked to sequenced samples.
    1.7. The paper reports that cohorts with sequenced samples are “suspected of failing”, “second-line” due to failure, “salvage”, or “children/adults failing first-line”.

2. The answer is likely to be “No” if :
    2.1. All sampled individuals are “ART/ARV/drug-naïve”, “before/at ART initiation”, “pre-ART/pre-treatment”, “newly diagnosed infections”, “primary infections”, “acute infections”, “blood donors”, or “transmitted resistance surveillance”.
    2.2. All sampled sequences are from virally suppressed on ART or elite controllers.

**Question 13: Were the patients in the study in a clinical trial?**

1. The answer is likely to be “Yes” if :
    1.1. The paper states “randomized”, “double-blind”, “placebo-controlled”, “open-label”, “controlled”, or “multicenter” with intervention arms.
    1.2. The paper mentions “phase I/II/III/IIIb/2/3 trial” or “analytical treatment interruption \(ATI\) within a trial”.
    1.3. The study is registered at ClinicalTrials.gov or has an NCT number.
    1.4. The participants were “enrolled in” or “participants in” named trials such as ACTG, HPTN 074/075/083, SPARTAC, DART/NORA, CAPRISA 004, DRIVE-FORWARD/DRIVE-AHEAD, GEMINI-1/2, DAWNING, CAPELLA, IMPAACT 2014, P1093, CHAPAS4, SEARCH, RV411, RESIST-2, FLAIR.
    1.5. The paper describes itself as a “substudy” of a clinical trial or “pre-randomization screening phase”.
    1.6. The paper reports a “registry” that explicitly has an NCT identifier indicating a trial e.g., PRESTIGIO Registry with NCT.

2. The answer is likely to be “No” if :
    2.1. The paper describes “observational”, “cohort” \(prospective/retrospective\), “cross-sectional”, “case-control”, “surveillance”, “epidemiological”, “national survey”, “sentinel surveillance”, or “routine clinical practice/testing/care”.
    2.2. Samples were from “outpatient/clinic/hospital cohorts”, “expanded access program \(EAP\)/compassionate use”, “registry/biobank”, or “screened by physicians”.
    2.3. The paper states explicitly that “patients in clinical trials were excluded”.

3. The answer is likely to be “Not Reported” if :
    3.1. The study is based on samples from named cohorts without trial designation such as ANRS CO5, ATHENA, InfCare HIV, HOMER, Sinikithemba, AIEDRP, RV217, UWPIC, FRESH, TARA.
    3.2. The paper focuses on case reports/series outside of clinical trials.
    3.3. Sequences/samples come from databases or reagent programs with mixed or unspecified sources \(e.g., HIVDB, GenBank, Los Alamos, NIAID Reagent Program\) and trial involvement isn’t specified.
    3.4. If the paper provides no statement about trial participation.

**Question 14: Does the paper report HIV sequences from individuals who had previously received ARV drugs?**
**If the answer to question 1 is “No”, then the answer to this question is “Not applicable”. Otherwise, read the remainder of the “if-then” rules for this question.**

1. The answer is likely to be “Yes” if :
    1.1. The paper uses phrases like “on ART/cART/HAART”, “ART-experienced”, “treatment-experienced”, “heavily pretreated”, “salvage regimen”, or “prior ARV exposure/regimens”.
    1.2. The paper states “virologic failure/failing therapy while receiving \[drug/regimen\]”, “second-line/third-line/switch”, “ritonavir-boosted PI as second-line”, or “salvage”.
    1.3. The paper states that patients received drugs belonging to a specific ARV class such as “NRTI”, “PI”, “NRTI”, “INSTI” and /or reports that patients received specific ARV drugs such as “3TC”, “FTC”, AZT”, “TDF”, “abacavir”, “efavirenz”, “rilpivirine”, “doravirine”, “atazanavir”, “lopinavir”, “darunavir”, “raltegravir”, “elvitegravir”, “dolutegravir”, “bictegravir”, “cabotegravir”, or “lenacapavir”.
    1.4. The paper reports that patients were virally suppressed on long-term ART.
    1.5. The paper states “genotyped at failure”, “samples collected during/after therapy”, “at virological failure \(VF\)”.
    1.6. The paper reports sequences from infants who had received ARV drugs or who had been exposed to drugs for PMTCT or maternal ART/prophylaxis.
    1.7. PrEP or ARVs are used in prevention trials with sequencing of viruses from study participants or seroconverters.
    1.8. The paper reports that sequencing was performed on cohorts that contain a mix of ART-naive and experienced persons.
    1.9. The paper reports that sequencing was performed on persons that were ART-experienced even if they were naïve to one or more drug classes.

2. The answer is likely to be “No” if :
    2.1. The paper states “ART-naive”, “treatment-naive”, “drug-naive”, “no previous ARV exposure”, “newly diagnosed/primary/acute/recent infection”, “baseline/pre-treatment”, or “before ART initiation/eligibility to start ART” at time of sampling.
    2.2. The paper only reports sequences from blood donors or persons undergoing surveillance for transmitted drug resistance in naive populations.
    2.3. The paper only reports elite or persistent controllers explicitly “without ART”.

**Question 15: Which drug classes were received by individuals in the study before sample sequencing?**
**If the answer to question 1 is “No”, then the answer to this question is “Not applicable”. Otherwise, read the remainder of the “if-then” rules for this question.**

1. If the paper says “treatment-naïve”, “ART-naïve”, “drug-naïve”, “never exposed to antiretrovirals”, “before ART initiation”, “pre-ART”, “newly diagnosed”, “seroconverter”, “blood donor”, “primary/acute infection without ART”, or samples collected “at diagnosis/baseline prior to therapy”, then return “None”.

2. If they are treatment experienced, they are almost certainly NRTI experienced.
    2.1. If they are from LMIC, prior to 2018, then almost always NRTI and NNRTI experienced.

3. If specific drugs or regimens are named, map to classes:
    3.1. NRTI: zidovudine/AZT, lamivudine/3TC, emtricitabine/FTC, tenofovir TDF/TAF, abacavir/ABC, stavudine/d4T, didanosine/ddI.
    3.2. NNRTI: efavirenz/EFV, nevirapine/NVP, etravirine/ETR, rilpivirine/RPV, doravirine/DOR.
    3.3. PI: lopinavir/r \(LPV/r\), atazanavir/ATV, darunavir/DRV, saquinavir/SQV, indinavir/IDV, nelfinavir/NFV, tipranavir/TPV, fosamprenavir/FPV, ritonavir boosting implies PI-based.
    3.4. INSTI: raltegravir/RAL, elvitegravir/EVG, dolutegravir/DTG, bictegravir/BIC, cabotegravir/CAB.
    3.5. CAI: lenacapavir/LEN.
    3.6. CCR5 antagonist: maraviroc/MVC.

4. If regimen descriptors are used, infer classes:
    4.1. “NNRTI-based first-line” implies NNRTI \+ NRTI.
    4.2. “PI-based second-line” implies PIs \+ NRTIs were administered to persons who previously received NRTIs \+ NNRTIs.
    4.3. “NRTI backbone” implies NRTIs were received.
    4.4. “TLD” refers to the NRTIs tenofovir and lamivudine \(3TC\) plus dolutegravir
    4.5. “Triomune” refers 3TC/d4T/NVP \(NRTI \+ NNRTI\).

5. For PrEP:
    5.1. TDF/FTC implies NRTIs.
    5.2. CAB-LA implies INSTI.
    5.3. CAB/RPV implies INSTI \+ NNRTI.
    5.4. LEN implies CAI.

6. If the paper states
    6.1. “On raltegravir/dolutegravir/elvitegravir/bictegravir” at sampling, include INSTI.
    6.2. “On maraviroc” include CCR5 antagonist.
    6.3. “On lenacapavir” include CAI.

7. If only “ART-experienced” is stated without drugs or class descriptors, return Not reported.

**Question 16: Which drugs were received by individuals in the study before sample sequencing?**
**If the answer to question 1 is “No” or if the answer to question 15 is “None”/ “Not Reported”/ “Not applicable”/,  then the answer to this question is “Not applicable”. Otherwise, read the remainder of the “if-then” rules for this question.**

1. If randomized regimens are described but sequencing was performed only at baseline before starting therapy, then report “None”

2. If only classes are given \(e.g., “two NRTIs \+ one NNRTI/PI”\) without drug names, then answer “Not Reported”.

3. If sampling is “at the time of virological failure”, “while receiving”, “on \[regimen\] at sampling”, “during \[drug\] treatment”, or “co-prescribed \[drug\]s”, then report those named drugs as received before sequencing.

4. If sequencing occurred after regimen start \(e.g., failure on first-line or salvage regimen\), then report all drugs in that regimen as received.

5. If integrase genotyping notes “INI-naïve” but lists concurrent NRTI/PI therapy, include only explicitly named NRTIs/PIs.

6. If switch/optimized background therapy lists drugs \(e.g., DRV/r, MVC, TDF/FTC\), include them.

7. If long-acting CAB/RPV used at sampling, consider CAB and RPV received.

8. Map synonyms/abbreviations:

AZT/ZDV \(zidovudine\), 3TC \(lamivudine\), FTC \(emtricitabine\), ABC \(abacavir\), D4T \(stavudine\), ddI \(didanosine\),
TDF \(tenofovir disoproxil fumarate\), TFV \(tenofovir\), TAF \(tenofovir alafenamide\), EFV \(efavirenz\), NVP
\(nevirapine\), ETR \(etravirine\), RPV \(rilpivirine\), DOR \(doravirine\), LPV/r \(lopinavir/ritonavir\), ATV/r
\(atazanavir/ritonavir\), DRV\(/r\) \(darunavir\), NFV \(nelfinavir\), SQV \(saquinavir\), IDV \(indinavir\), FPV
\(fosamprenavir\), TPV \(tipranavir\), RTV \(ritonavir\), RAL \(raltegravir\), EVG \(elvitegravir\), DTG \(dolutegravir\), BIC
\(bictegravir\), CAB \(cabotegravir\), LEN \(lenacapavir\), MVC \(maraviroc\), IBA \(ibalizumab\)
