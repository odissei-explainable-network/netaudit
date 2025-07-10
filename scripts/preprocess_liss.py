"""Preprocess LISS panel data for predictive modeling.
"""

import argparse
import logging
import os


import polars as pl

logger = logging.getLogger(__name__)


def cmdline_parse_args(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Preprocess LISS panel data for predictive modeling.",
        epilog="""Example: python preprocess_liss.py
"""
    )

    parser.add_argument("--liss-filename", default="H:/data/raw/9780liss_data_import_20241003CBKV1.csv", help="Filename of imported LISS panel data including Personality and Politics and Values modules.")
    parser.add_argument("--link-filename", default="H:/data/raw/LISSkoppelbestandrespondenten2024V3.csv", help="Filename of linkage keys to link LISS panel subjects to CBS identifiers.")
    parser.add_argument("--gbapersoon-filename", default="G:/Bevolking/GBAPERSOONTAB/2022/geconverteerde data/GBAPERSOON2022TABV2.csv", help="Filename of demographic background file (GBAPERSOONTAB).")
    parser.add_argument("--income-filename", default="H:/data/raw/INPA2022TABV2.csv", help="Filename of income data file (INPATAB).")
    parser.add_argument("--education-filename", default="G:/Onderwijs/HOOGSTEOPLTAB/2022/geconverteerde data/HOOGSTEOPL2022TABV1.csv", help="Filename of highest achieved education file (HOOGSTEOPLTAB).")
    parser.add_argument("--out-filename", default="H:/data/processed/liss_preprocessed.csv", help="Filename of preprocessed output file.")
    return parser.parse_args(args)


def format_polars_df(df, precision=2):
    """Helper function for printing polars dataframe during logging.
    """
    rows = []

    col_widths = {col: max(len(str(col)), max(len(str(val)) for val in df[col].to_list())) for col in df.columns}

    header = " | ".join(str(col).ljust(col_widths[col]) for col in df.columns)

    separator = "-" * len(header)

    rows.append(header)
    rows.append(separator)

    for row in df.iter_rows():
        formatted_row = []
        for col, val in zip(df.columns, row):
            if isinstance(val, float):
                formatted_val = f"{val:.{precision}f}"
            else:
                formatted_val = str(val)
            formatted_row.append(formatted_val)
        rows.append(" | ".join(formatted_row))

    return "\n".join(rows)


def main(args=None):
    args = cmdline_parse_args(args)

    logging.basicConfig(
        handlers=[
            logging.FileHandler(
                os.path.join("logs", "preprocess_liss.log"),
                mode="w"
            ),
            logging.StreamHandler()
        ],
        format="%(asctime)s - [%(levelname)s] - %(message)s",
        level=logging.INFO,
        encoding="utf-8"
    )

    logger.info("Reading LISS file from %s", args.liss_filename)
    df_liss = pl.read_csv(args.liss_filename, separator=";", null_values=["NA", "-9", "-8"]) #data, I don't know (-9) and I prefer not to say (-8) are treated as missing
    
    logger.info("Reading LISS link file from %s", args.link_filename)
    df_link = (pl.read_csv(args.link_filename, separator=";", null_values=[" "])
        .select(
            pl.col("nomem_encr_crypt").alias("nomem_encr_crypt"),
            pl.col("RINPERSOON").cast(pl.Int64)
        )
    )

    logger.info("Raw link data has %s missing RINPERSOON rows", df_link.select(pl.col("RINPERSOON").is_null().sum()).item())

    logger.info("Reading GBAPERSOON file from %s", args.gbapersoon_filename)
    df_gbapersoon = (pl.scan_csv(args.gbapersoon_filename, separator=";", null_values=["", "-", "----", "-1", "Onbekend"]) # Unknown is treated as missing
        .filter(pl.col("RINPERSOONS") == "RINPERSOON WEL IN GBA/BRP") # Only select persons in GBA/BRP
        .select(["RINPERSOON", "GBAGEBOORTEJAAR", "GBAGESLACHT", "GBAAANTALOUDERSBUITENLAND"]) # Only select relevant columns
        .unique("RINPERSOON") # Make sure we have unique person identifiers
        .collect()
    )

    logger.info("Reading INPA file from %s", args.income_filename)
    # -1 = Without personal income, -2 = Person in household without registered income, -3 = Person in insititutional household
    # 9999999999 = Person does not belong to target population of purchasing power calculation
    df_inpatab = (pl.scan_csv(args.income_filename, separator=";", null_values=["-", "----", "-1", "-2", "-3", "9999999999"], schema_overrides={"RINPERSOON": pl.Int64, "RINPERSOONHKW": pl.Int64})
        .filter(pl.col("RINPERSOONS") == "R") # Only persons in GBA/BRP
        .select(["RINPERSOON", "INPKKGEM", "INPKKMUT", "INPP100PBRUT"])
        .unique("RINPERSOON")
        .collect()
    )

    logger.info("Reading HOOGSTEOPL file from %s", args.income_filename)
    # -1, 9999 = Unknown
    df_hoogsteopltab = (pl.scan_csv(args.education_filename, separator=";", null_values=["-", "----", "-1", "9999"], schema_overrides={"RINPERSOON": pl.Int64})
        .filter(pl.col("RINPERSOONS") == "R")
        .select(["RINPERSOON", "OPLNIVSOI2021AGG4HBmetNIRWO"])
        .unique("RINPERSOON")
        .collect()
    )

    gender_dict = {
        "Vrouwen": "female",
        "Mannen": "male",
    }

    parents_abroad_dict = {
        "geen ouders in het buitenland": "zero",
        "1 ouder in het buitenland": "one",
        "beide ouders in het buitenland": "both",
    }

    education_dict = {
        "11": "primary",
        "12": "vocational",
        "21": "higher_vocational",
        "31": "bachelor",
        "32": "master_doctorate"
    }

    df_liss_linked = (df_liss.join(
            df_link.filter(pl.col("RINPERSOON").is_not_null()), #IDs. ~5000 out of 7000 match to RINPERSOON
            how="inner",
            validate="1:1",
            on="nomem_encr_crypt"
        )
        .join(df_gbapersoon, how="left", validate="1:1", on="RINPERSOON")
        .join(df_inpatab, how="left", validate="1:1", on="RINPERSOON")
        .join(df_hoogsteopltab , how="left", validate="1:1", on="RINPERSOON")
        .select([
                "RINPERSOON",
                pl.col("cv24p053").alias("cv24p053_voted"), #1=yes, 2=no, 3=not eligible
                pl.col("cv24p307").alias("cv24p307_voting_behavior"), #1=vvd, 2=pvv, 3=cda, 4=d66, 6=sp, 8=cu, 9=pvdd, 11=sgp, 12=denk, 13=fvd, 14=blank, 15=other, 16=volt, 17=ja21, 18=bbb, 20=gl/pvda, 21=nsc
                pl.col("cv24p307").is_in([2, 13, 17, 18]).alias("cv24p307_voting_behavior_populist"),  #2=PVV, 13=FVD, 17=JA21, 18=BBB
                pl.col("cv24p013").alias("cv24p013_trust_government"), #0 no confidence to 10 full confidence
                pl.col("cp24p010").alias("cp24p010_happiness"), #0 totally unhappy to 10 totally happy
                pl.col("cp24p012").alias("cp24p012_feel_state"), #1 very bad to 7 very good
                pl.col("cp24p013").alias("cp24p013_feel_trait"), #1 very bad to 7 very good
                pl.col("cp24p014").alias("cp24p014_satisfaction"), #0 not at all satisfied to 10 completely satisfied
                pl.col("cp24p015").alias("cp24p015_satisfaction"), #0 not at all satisfied to 10 completely satisfied
                pl.col("cp24p016").alias("cp24p016_satisfaction"), #0 not at all satisfied to 10 completely satisfied
                pl.col("cp24p017").alias("cp24p017_satisfaction"), #0 not at all satisfied to 10 completely satisfied
                pl.col("cp24p018").alias("cp24p018_satisfaction"), #0 not at all satisfied to 10 completely satisfied
                pl.col("cp24p019").alias("cp24p019_interpersonal_trust"), #0 not at all satisfied to 10 completely satisfied
                (pl.col("cp24p020") - pl.col("cp24p025") + pl.col("cp24p030") - pl.col("cp24p035") + pl.col("cp24p040") - pl.col("cp24p045") + pl.col("cp24p050") - pl.col("cp24p055") + pl.col("cp24p060") - pl.col("cp24p065")).alias("cp24p020+_extraversion"),
                (-(pl.col("cp24p021") - pl.col("cp24p026") + pl.col("cp24p031") - pl.col("cp24p036") + pl.col("cp24p041") - pl.col("cp24p046") + pl.col("cp24p051") - pl.col("cp24p056") + pl.col("cp24p061") - pl.col("cp24p066"))).alias("cp24p021+_agreeableness"),
                (pl.col("cp24p022") - pl.col("cp24p027") + pl.col("cp24p032") - pl.col("cp24p037") + pl.col("cp24p042") - pl.col("cp24p047") + pl.col("cp24p052") - pl.col("cp24p057") + pl.col("cp24p062") - pl.col("cp24p067")).alias("cp24p022+_conscientiousness"),
                (-(pl.col("cp24p023") - pl.col("cp24p028") + pl.col("cp24p033") - pl.col("cp24p038") + pl.col("cp24p043") - pl.col("cp24p048") + pl.col("cp24p053") - pl.col("cp24p058") + pl.col("cp24p063") - pl.col("cp24p068"))).alias("cp24p023+_emotional_stability"),
                (pl.col("cp24p024") - pl.col("cp24p029") + pl.col("cp24p034") - pl.col("cp24p039") + pl.col("cp24p044") - pl.col("cp24p049") + pl.col("cp24p054") - pl.col("cp24p059") + pl.col("cp24p064") - pl.col("cp24p069")).alias("cp24p024+_imagination"),
                (pl.col("cp24p070") + pl.col("cp24p071") - pl.col("cp24p072") + pl.col("cp24p073") - pl.col("cp24p074") + pl.col("cp24p075") + pl.col("cp24p076") - pl.col("cp24p077") - pl.col("cp24p078") - pl.col("cp24p079")).alias("cp24p070+_self_esteem"), #higher=more self-esteem
                pl.col("cp24p135").alias("cp24p135_closeness"), #1 = not close, 7 = very close 
                (pl.col("cp24p136") + pl.col("cp24p137") - pl.col("cp24p138") - pl.col("cp24p139") + pl.col("cp24p140") + pl.col("cp24p141") - pl.col("cp24p142") + pl.col("cp24p143") - pl.col("cp24p144") - pl.col("cp24p145")).alias("cp24p136+_social_desirability"), #higher=more social desirability. careful with interpretation
                (pl.col("cp24p198") - pl.col("cp24p200") + pl.col("cp24p201") - pl.col("cp24p204") - pl.col("cp24p206") + pl.col("cp24p207")).alias("cp24p198+_optimism"), #higher=more optimism. careful with interpretation, test starts on 0, liss starts on 1
                (pl.lit(2024) - pl.col("GBAGEBOORTEJAAR")).alias("cbs_age"),
                pl.col("GBAGESLACHT").replace_strict(gender_dict).alias("cbs_gender_2_cat"),
                pl.col("GBAAANTALOUDERSBUITENLAND").replace_strict(parents_abroad_dict).alias("cbs_num_parents_abroad_3_cat"),
                pl.col("INPKKGEM").cast(pl.Float64).log1p().alias("cbs_purch_power_log"),
                pl.col("INPKKMUT").alias("cbs_change_purch_power"),
                pl.col("INPP100PBRUT").alias("cbs_gross_income_percentile"),
                pl.col("OPLNIVSOI2021AGG4HBmetNIRWO").cast(pl.String).str.slice(0, 2).replace_strict(education_dict).alias("cbs_high_achieved_edu_degree_5_cat") #11=primary school, #12=vocational school, #21=higher vocational school, #31=bachelor's degree, #32=master's or doctoral degree
            ])
    )

    logger.info("Raw liss data has %s rows", df_liss.shape[0])
    logger.info("Linked liss data has %s rows", df_liss_linked.shape[0])

    voting_behavior_dict = {-2: "not_elligible", -1: "not_voted", 1: "vvd", 2: "pvv", 3: "cda", 4: "d66", 6: "sp", 8: "cu", 9: "pvdd", 11: "sgp", 12: "denk", 13: "fvd", 14: "blank", 15: "other", 16: "volt", 17: "ja21", 18: "bbb", 20: "gl/pvda", 21: "nsc"}

    populist_voting_behavior_dict = {
        -2: "not_elligible", 
        -1: "not_voted",
        0: "non_populist",
        1: "populist",
    }

    df_liss_linked = (df_liss_linked
        .with_columns(
            (pl.when(pl.col("cv24p053_voted").eq(2)).then(-1) # -1=not voted
                .when(pl.col("cv24p053_voted").eq(3)).then(-2) # -2=not elligible
                .otherwise(pl.col("cv24p307_voting_behavior"))
                .alias("cv24p307_voting_behavior")
                .replace_strict(voting_behavior_dict)
                .fill_null("missing")
            ),
            (pl.when(pl.col("cv24p053_voted").eq(2)).then(-1) # -1=not voted
                .when(pl.col("cv24p053_voted").eq(3)).then(-2) # -2=not elligible
                .otherwise(pl.col("cv24p307_voting_behavior_populist"))
                .alias("cv24p307_voting_behavior_populist")
                .replace_strict(populist_voting_behavior_dict)
                .fill_null("missing")
            )
        )
    )

    logger.info("Linked liss data has %s missing voting outcome cases", df_liss_linked.select((pl.col("cv24p307_voting_behavior").eq("missing")).sum().alias("count_missing")).item())

    logger.info("Linked liss data unique counts for 'voted'\n%s", format_polars_df(df_liss_linked.select(pl.col("cv24p053_voted").value_counts()).unnest("cv24p053_voted")))
    logger.info("Linked liss data unique counts for 'voting_behavior'\n%s", format_polars_df(df_liss_linked.select(pl.col("cv24p307_voting_behavior").value_counts()).unnest("cv24p307_voting_behavior")))
    logger.info("Linked liss data unique counts for 'voting_behavior_populist'\n%s", format_polars_df(df_liss_linked.select(pl.col("cv24p307_voting_behavior_populist").value_counts()).unnest("cv24p307_voting_behavior_populist")))
    logger.info("Linked liss data unique counts for 'trust_government'\n%s", format_polars_df(df_liss_linked.select(pl.col("cv24p013_trust_government").value_counts()).unnest("cv24p013_trust_government")))

    logger.info("Linked liss missing counts for 'voting_behavior' and 'trust_government'\n%s", format_polars_df(df_liss_linked.select(pl.col("cv24p307_voting_behavior").eq("missing").alias("voting_behavior_missing"), pl.col("cv24p013_trust_government").is_null().alias("trust_government_missing")).group_by(["voting_behavior_missing", "trust_government_missing"]).count()))

    df_liss_linked.write_csv(args.out_filename, separator=",")

    logger.info("Wrote linked file to %s", args.out_filename)


if __name__ == '__main__':
	main()
        