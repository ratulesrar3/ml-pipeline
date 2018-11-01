import psycopg2
import numpy as np
import pandas as pd
import json
from datetime import timedelta
from setup import Connecc


ALL_BOOKINGS = '''
                DROP TABLE IF EXISTS ab;
                CREATE TEMP TABLE ab AS (
                SELECT hash_ssn, booking.mni_no, dob, race, sex, city, state, zip, country,
            			us_zip_first_five, tract2010id, blockgroup2010id, block2010id, joco_resident,
            			booking.case_no, booking.case_type, booking.booking_no,
            			booking.arresting_agency, booking.arresting_agency_type,
                        booking.booking_date, booking.release_date,
            			booking.bail_type, booking.bail_amt, booking.bailed_out
            	FROM clean.jocojimsperson_hashed indiv
            	JOIN clean.jocojimsjailbooking_hashed booking
            	ON indiv.mni_no::int = booking.mni_no::int
                WHERE hash_ssn IS NOT NULL AND release_date IS NOT NULL
            	);
                '''
BOOKING_MH = '''
            DROP TABLE IF EXISTS bk_mh;
            CREATE TEMP TABLE bk_mh AS (
	           SELECT ab.hash_ssn, ab.mni_no, ab.dob, ab.race, ab.sex, ab.city, ab.state, ab.zip, ab.country,
        			ab.us_zip_first_five, ab.tract2010id, ab.blockgroup2010id, ab.block2010id, ab.joco_resident,
        			ab.case_no, ab.case_type, ab.booking_no, ab.arresting_agency, ab.arresting_agency_type,
        			ab.booking_date, ab.release_date, ab.bail_type, ab.bail_amt, ab.bailed_out,
        			mh.admit_date, mh.dschrg_date, mh.program, mh.pri_dx_code, mh.pri_dx_value,
        			mh.refferal_source referral_src, mh.case_id mh_case_id
        	FROM ab
        	LEFT JOIN clean.jocomentalhealth_hashed mh
        	ON ab.hash_ssn = mh.hash_ssn
        	);
            '''


def create_tables(connection):
    connection.execute_query(ALL_BOOKINGS)
    connection.execute_query(BOOKING_MH)


def get_jail_hist(range_before_jail):
    '''
    returns query to obtain jail history
    for a given interval before booking
    '''

    case_query = []
    for ctype in ['DV', 'JV', 'CR']:
        q = '''
            SUM(CASE WHEN priors.case_type = '{}' THEN 1 ELSE 0 END) AS {}_count
            '''.format(ctype, ctype)
        case_query.append(q)
    case_query = ','.join(case_query)

    jail_hist = '''
        SELECT rel.hash_ssn, rel.release_date,
        (CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END) AS jail_hist,
        {},
        AVG(EXTRACT(DAY FROM(priors.release_date - priors.booking_date))::int) avg_jail_time,
        SUM(EXTRACT(DAY FROM(priors.release_date - priors.booking_date))::int) total_jail_time,
        SUM(CASE WHEN priors.bailed_out = 'true' THEN 1 ELSE 0 END) AS past_bail_hist
        FROM bk_mh priors
        JOIN bk_mh rel ON priors.hash_ssn = rel.hash_ssn AND priors.release_date < rel.release_date
        WHERE priors.release_date BETWEEN(rel.booking_date - INTERVAL '{}') AND rel.booking_date
        GROUP BY rel.hash_ssn, rel.release_date
        '''.format(case_query, range_before_jail)

    return jail_hist


def get_mh_hist(range_before_jail):
    '''
    returns query to obtain mental health history (count of admit dates)
    for a given interval before booking
    '''
    ref_srcs = ['SELF', 'COMMUNITY SERVICE PROGRAMS', 'RESIDENTIAL FACILITY (ADULT GRP HOME)', 'SOCIAL/COMMUNITY AGENCY',
	'S.R.S. - AREA OFFICE', 'ALC/DRUG TREATMENT PROGRAM', 'COMMUNITY DEVELOPMENTAL DISABILITY ORGANIZATION', 'CRISIS STABILIZATION UNIT',
    'FAMILY & RELATIVES', 'FRIENDS', 'CLERGY', 'COMMUNITY CORRECTIONS', 'PENAL SYSTEM', 'DUI/DWI', 'POLICE',  'COURT',
	'JUV.CORRETION FACILITY(TJCF, LJCF, AJCF, BJCF)', 'PROBATION', 'COMMUNITY CORRECITONS', 'DIVERSIONARY PROGRAM', 'JUVENILE JUSTICE AUTHORITY', 'ATTORNEY',
    'SCHOOL/COLLEGE EXCLUDING LOCAL PRIM/SEC', 'EMPLOYEE ASSISTANCE PGM(EAP):INCL EMPL REFERRAL', 'LOCAL SCHOOL (PRIM/SEC EDUC)', 'ADOPTION PRIVITIZATION CONTRACT', 'FOSTER CARE PRIVATIZATION CONTRACT',
	'FOSTER CARE/FAMILY PRESERVATION', 'YOUTH RESIDENTIAL GROUP HOME', 'PRIVATE PSYCH HOSPITAL', 'STATE OR LOCAL HEALTH DEPARTMENT', 'OTHER PRIVATE HEALTH CARE PROFESSIONAL',
	'NURSING FACILITY', 'MENTAL HEALTH CONSORTIUM (KS)', 'GENERAL HOSPITAL', 'MANAGED CARE ORGANIZATION', 'STATE MENTAL HEALTH HOSPITAL',
	'MENTAL HEALTH CENTER', 'PRIMARY CARE PHYSICIAN', 'MILITARY', 'NO ENTRY', 'IN HOUSE STAFF', 'UNKNOWN']

    programs = ['PEDIATRICS MH PROGRAM', 'CSS PGM', 'ADULT MH PROGRAM', 'FF STEP AHEAD', 'PRE ADMIT HOSPITAL SCREENING',
    'ADMISSION PROGRAM', 'FORENSIC', 'FAMILY FOCUS', 'OUTPATIENT', 'PRE ADMIT EVALUATION', 'DEAF SERV-OPT', 'SEXUAL ABUSE',
    'SUP EMPLOYMENT', 'VOCATIONAL SERVICES', 'MEDICATION ONLY', 'PRE ADMIT OUTREACH', 'EMERGENCY SRV', 'FF OUTPATIENT']

    y = range_before_jail[0]
    cols_to_select = {}

    ref_query = []
    strip_chars = [" ", ",", ":", "/", "\\", "&", ".", "-", "(", ")"]
    for ref_type in ref_srcs:
        ref_type_clean = ref_type
        for i in strip_chars:
            ref_type_clean = ref_type_clean.replace(i, "")
        col_name = "ref_{}_{}_yr".format(ref_type_clean, y)
        r = '''
            SUM(CASE WHEN bk_mh.referral_src = '{}' THEN 1 ELSE 0 END) AS {}
            '''.format(ref_type, ref_type_clean)
        ref_query.append(r)
        cols_to_select[ref_type_clean] = col_name

    ref_query = ','.join(ref_query)

    prog_query = []
    for prog_type in programs:
        prog_type_clean = prog_type
        for i in strip_chars:
            prog_type_clean = prog_type_clean.replace(i, "")
        col_name = "prog_{}_{}_yr".format(prog_type_clean, y)
        p = '''
            SUM(CASE WHEN bk_mh.program = '{}' THEN 1 ELSE 0 END) AS {}
            '''.format(prog_type, prog_type_clean)
        prog_query.append(p)
        cols_to_select[prog_type_clean] = col_name

    prog_query = ','.join(prog_query)

    diagnoses = ['PSYCHOSIS', 'DEPRESSIVE', 'ANXIETY', 'BIPOLAR', 'SCHIZOPHRENIA']
    diag_query = []
    for diag in diagnoses:
        col_name = "{}_{}_yr".format(diag, y)
        d = '''
            SUM(CASE WHEN bk_mh.pri_dx_value LIKE '%{}%' THEN 1 ELSE 0 END) AS {}
            '''.format(diag, diag)
        diag_query.append(d)
        cols_to_select[diag] = col_name
    diag_query = ','.join(diag_query)

    mh_hist = '''
        SELECT hash_ssn, (CASE WHEN COUNT(admit_date) > 0 THEN 1 ELSE 0 END) AS mh_hist,
        AVG(EXTRACT(DAY FROM(dschrg_date - admit_date))::int) avg_mh_time,
        SUM(EXTRACT(DAY FROM(dschrg_date - admit_date))::int) total_mh_time,
        {}, {}, {} FROM bk_mh
        WHERE admit_date BETWEEN(booking_date - INTERVAL '{}') AND booking_date
        GROUP BY hash_ssn
        '''.format(ref_query, prog_query, diag_query, range_before_jail)

    cols = []
    for k, v in cols_to_select.items():
        cols.append('mh{}.{} AS {}'.format(y, k, v))

    mh_cols = ','.join(cols)

    return mh_hist, mh_cols


def get_label(interval):
    '''
    returns query to obtain mental health entry within given interval after release
    '''

    label = '''
            SELECT hash_ssn, release_date, min(admit_date) AS first_mh,
                (CASE WHEN min(admit_date) < release_date + INTERVAL '{}'
                AND MIN(admit_date) > release_date THEN 1 ELSE 0 END) AS label
            FROM bk_mh
            GROUP BY hash_ssn, release_date
            '''.format(interval)
    return label


def get_lsir():
    lsir_vars = ['lsir_no', 'case_no', 'percentile', 'total_score', 'cls', 'geom']

    temp = "SELECT {} FROM public.jocojims2lsirdata".format(",".join(lsir_vars))
    select_lsir = []
    for col in lsir_vars:
        if col != 'case_no':
            select_lsir.append("lsir.{} as lsir_{}".format(col, col))
    select_lsir = ','.join(select_lsir)
    return temp, select_lsir


def join_queries(cutoff_date, test, validation_date):
    '''
    returns queries to get features for train or test set
    '''

    mh_hist_1_yr, mh1_select = get_mh_hist('1 year')
    mh_hist_3_yr, mh3_select = get_mh_hist('3 year')
    label_1_yr = get_label('1 year')
    jail_hist_1_yr = get_jail_hist('1 year')
    jail_hist_3_yr = get_jail_hist('3 year')

    demo = '''
            SELECT DISTINCT hash_ssn, release_date, sex, race,
                (EXTRACT(DAY FROM(release_date - dob))/365.25)::int AS age,
                EXTRACT(DAY FROM dob) AS bday,
                EXTRACT(MONTH FROM dob) AS bmonth,
                city, state, us_zip_first_five, tract2010id,
                joco_resident, arresting_agency, arresting_agency_type,
                bail_type, bail_amt, bailed_out, case_type, booking_date,
                EXTRACT(MONTH FROM booking_date) AS booking_month,
                EXTRACT(DAY FROM(release_date - booking_date))::int AS jail_time,
                case_no
            FROM bk_mh
            '''
    lsir_temp, lsir_select = get_lsir()

    query = '''
            WITH demo AS ({}),
            label AS ({}),
            mh1 AS ({}),
            mh3 AS ({}),
            jh1 AS ({}),
            jh3 AS ({}),
            lsir AS ({})

            SELECT demo.hash_ssn, demo.release_date, demo.sex, demo.age, demo.race,
            demo.city, demo.state, demo.us_zip_first_five, demo.tract2010id,
            demo.bday, demo.bmonth, demo.bail_type, demo.bail_amt, demo.bailed_out,
            demo.case_type, demo.jail_time, demo.booking_date, demo.booking_month,
            demo.joco_resident, demo.arresting_agency, demo.arresting_agency_type,
            mh1.mh_hist AS mh_1yr, mh1.avg_mh_time mh_avg_time_1yr, mh1.total_mh_time mh_total_time_1yr, {},
            mh3.mh_hist AS mh_3yr, mh3.avg_mh_time mh_avg_time_3yr, mh3.total_mh_time mh_total_time_3yr, {},
            jh1.jail_hist AS jail_1yr, jh1.avg_jail_time AS jail_avg_time_1_yr, jh1.total_jail_time AS jail_total_time_1_yr,
            jh1.dv_count AS dv_1yr, jh1.cr_count AS crim_1yr, jh1.jv_count AS juv_1yr, jh1.past_bail_hist AS bail_1yr,
            jh3.jail_hist AS jail_3yr, jh3.avg_jail_time AS jail_avg_time_3_yr, jh3.total_jail_time AS jail_total_time_3_yr,
            jh3.dv_count AS dv_3yr, jh3.cr_count AS crim_3yr, jh3.jv_count AS juv_3yr, jh3.past_bail_hist AS bail_3yr,
            {},
            label.label

            FROM demo
            LEFT JOIN label ON demo.hash_ssn = label.hash_ssn AND demo.release_date = label.release_date
            LEFT JOIN mh1 ON demo.hash_ssn = mh1.hash_ssn
            LEFT JOIN mh3 ON demo.hash_ssn = mh3.hash_ssn
            LEFT JOIN jh1 ON demo.hash_ssn = jh1.hash_ssn AND demo.release_date = jh1.release_date
            LEFT JOIN jh3 ON demo.hash_ssn = jh3.hash_ssn AND demo.release_date = jh3.release_date
            LEFT JOIN lsir ON demo.case_no = lsir.case_no
            '''.format(demo, label_1_yr, mh_hist_1_yr, mh_hist_3_yr,
                    jail_hist_1_yr, jail_hist_3_yr, lsir_temp,
                    mh1_select, mh3_select, lsir_select)

    if test:
        query += "WHERE demo.release_date BETWEEN '{}' AND '{}'".format(cutoff_date, validation_date)
    else:
        query += "WHERE demo.release_date < '{}'".format(cutoff_date)

    return query


def create_train_test(connection, cutoff_date, validation_date):
    '''
    returns train test sets based on cutoff and validation date
    '''

    train_q = join_queries(cutoff_date, False, validation_date)
    test_q = join_queries(cutoff_date, True, validation_date)

    train = connection.get_df(train_q)
    test = connection.get_df(test_q)

    return train, test


def generate_features(connection):
    '''
    get data from db and
    generate pickle files for each time split
    '''

    create_tables(connection)

    cv = [('2010-12-31', '2011-12-31'),
          ('2011-12-31', '2012-12-31'),
          ('2012-12-31', '2013-12-31'),
          ('2013-12-31', '2014-12-31'),
          ('2014-12-31', '2015-12-31')]

    for c, v in cv:
        train, test = create_train_test(connection, c, v)
        train.to_pickle('data/c{}_v{}_train.pkl'.format(c[:4], v[:4]))
        test.to_pickle('data/c{}_v{}_test.pkl'.format(c[:4], v[:4]))


def go_ft():
    nu_connecc = Connecc()
    nu_connecc.set_connection()
    generate_features(nu_connecc)
