# Approach Developed by Grant Lewis
# Updated Oct. 4, 2022

import json
import os
from tqdm import tqdm
import pandas as pd
import re
import sys


SHOULD_BATCH_SAVE = True
BATCH_SIZE = 100000 # 5000

PRINT_SAVE_COUNTS = True
SHOULD_HIDE_PROGRESS_BAR = False


POST_NAME = "abortion_posts.csv"
COMMENT_NAME = "abortion_comments.csv"

SET_POST_AS_PARENT = False

# Example Command: python ./abortion_csv_builder.py ./Jan_March_2019/ ./test_csvs_2/


ACCEPTED_FILE_TYPES = tuple([".json"]) # can add more values to the list

HYPERLINK_SEARCH = r"[^\[](?:(https?://)|(/?[ru]/[A-Za-z0-9\-_]))" # identify hyperlinks (not including those inside of square brackets (to avoid duplicate counting))

GROUP_NAMEING = "flair_{}"
UNKNOWN_GROUP = "unknown"
NONE_GROUP = "none"
# (group, regex) # Order Matters!
GROUPS = [("unverified", r"(?i:layperson|not .*verified)"),
        ("physician", r"(?i:physician(?!s? *assistant))"),
        ("mid_level_provider", r"(?i:Physicians? Assistant(?! - )|Nurs.* (Pract|doc|anes)|(?:NP|PA)[ -](?!stu)|(?<!para)medic(?:[^a-z]|$))"),
        ("nurse", r"(?i:nurs(?:e|ing)(?! *(pract|doct|anesth))|(?:[^a-z]|^)(?:C{0,2}RN|CNA)( |$)|LPN|BSN|NP stu)"),
        ("medical_student", r"(?i:(^| )(med|MD).*(school|stt?udent?))"),
        ("other", r"(?i:web developer|helpful bot)"),
        ("technician", r"(?i:(Lab|tech|EMT|Paramed|Emerg|medical *assistant|first responder|graph|perfusion|path.* assistant))"),
        ("allied_health", r"(?i:clinic|psycho|physio|counsel|thera|pharm(?!acy technician)|biomed|chiro|midwife|speech|social|slp|( |^)dent|public|epidem|podi|bio(?:log|chem|behav)|dieti|dont|Care (?:coord|as)|(?<!histo)patho|research|theatre|ot(?!olar)|scientist|librar|coder|communication|athe?letic|prosthetist|specialist|opto|audio|scribe|mental health|LCSW|DPT Student)"),
        ("physician_2", r"(?i:(^|\W)ent(\W|$)|(^|\W)Doctor(\W|$)|surge(on|ry)|peds|pediatric(?! nurse)|((?<!techn|psych| path|idemi)olog|(?<!podia)tr)ist|M\.?D\.?|Emergency Physician|(?<!nurse )practitioner|geriatrician|resident)"),
        ("administration", r"(?i:(?<!\| )(?:Moderator|Founder))"),
        (NONE_GROUP, r"^(?![\s\S])"),
        (UNKNOWN_GROUP, r"")
        ]

GROUPS = [(GROUP_NAMEING.format(n), re.compile(v)) for (n,v) in GROUPS]
UNKNOWN_GROUP = GROUP_NAMEING.format(UNKNOWN_GROUP)
NONE_GROUP = GROUP_NAMEING.format(NONE_GROUP)
print(GROUPS)





# Internal/finetune settings
# Post Columns
PC_POST_ID = "post_id"
PC_SUBREDDIT = "subreddit"
PC_TIME = "created_utc"
PC_TIME_TO_COMMENT = "time_to_first_comment"
PC_TITLE = "title"
PC_BODY = "selftext"
PC_BODY_LEN = "text_length"
PC_COMMENTS = "comments"
PC_COMMENT_COUNT = "num_comments"
PC_AUTHOR = "author"
PC_AUTHOR_FLAIR = "author_flair"
PC_LINK_COUNT = "hyperlink-count"
PC_LINK_FLAIR = "link_flair_text"
PC_AUTHOR_FLAIR_GROUP = "author_flair_group_name"
PC_RACE = "author_race"
PC_GENDER = "author_gender"

PC_TO_DROP = ["thumbnail", "media", "is_reddit_media_domain",
                    "is_video", "is_original_content"]
PC_TO_DROP.append(PC_COMMENTS)

# Comment Columns
CC_COMMENT_ID = "comment_id"
CC_SUBREDDIT = "subreddit"
CC_PARENT_ID= "parent_id" if SET_POST_AS_PARENT else "parent_comment_id"
CC_POST_ID = PC_POST_ID
CC_OTHERS_FROM_POST = []
CC_TIME = PC_TIME
CC_TIME_TO_NESTED = "time_to_first_nested_comment"
CC_BODY = "body"
CC_BODY_LEN = "text_length"
CC_NESTED_COMMENTS = "children"
CC_NESTED_COUNT = "num_nested_comments"
CC_AUTHOR = "author"
CC_AUTHOR_FLAIR = "author_flair"
CC_AUTHOR_FLAIR_GROUP = "author_flair_group_name"
CC_LINK_COUNT = "hyperlink-count"

CC_TO_DROP = []
CC_TO_DROP.append(CC_NESTED_COMMENTS)


AUTO_MOD_AUTHOR_NAMES = ["AutoModerator"] # Auto Moderator identifiers
NONE_VALUE = "null" # string to fill a None value with
RAISE_COMMENT_OVERWRITE = True # Raise an error if a key already exists in the comment dict when trying to add new columns (helps prevent against overwritting data)  


# For Testing Purposes:
RUN_PARSER = True
SHOULD_MOD_NEWLINES = True#False # replace \n and \t with \\n and \\t respectively
TEST_READ = False


# For Author Gender and Race
AUTH_SECTION_LOCATION_L1 = re.compile(r"(?i:(stats|details|aged?|sex|gender|race|height?|weight?|locations?|duration(?: of complaint)?|(?:current )?medications?| i'? ?a?m)[^a-z0-9])")
AUTH_SECTION_LOCATION_L2 = re.compile(r"(?i:(?:(?:\d{1,3}[^a-z0-9]*(?:m|f|[mf]to?[mf]))|(?:(?:m|f|[mf]to?[mf])[^a-z0-9]*\d{1,3}))[^a-z0-9])")
AUTH_SECTION_LOCATION_L3 = re.compile(r'(?i:smoke|drink|drugs|meds|\d[^a-z0-9]*(?:(?:lb|kg|pound|ounce)s?|ft|foot|feet|c?m|(?:centi)?meters?|tall|(?:year|yr)s?([^a-z0-9]old?)|""|\'\d)|(?:fe)?male)|medical history|(?:other )info(?:rmation)[^a-z0-9]')

BAD_TEXT_POSTS = re.compile(r"(?i:\[(?:removed|deleted)\])")

AUTH_LAYERS = [(AUTH_SECTION_LOCATION_L1, 3), (AUTH_SECTION_LOCATION_L2, 2), (AUTH_SECTION_LOCATION_L3, 1)]

AUTH_DISTANCE_BUFFER = 25 #20#15#10 #0.05 # > 1 = number of chars from start; < 1 = % of chars from start

AUTH_NONE = "Empty" #"Remove"
AUTH_UNSPECIFIED = "Unspecified"
AUTH_CONFLICT = "Conflict"

AUTH_CHOOSE_FIRST = False
AUTH_SELECT_BY_WEIGHT = True

AUTH_INTERESTS = {'sex':
                {'re': re.compile(r'(?i:((?:(?:(?:(?:(?:fe)?male(?: *to *(?:fe)?male)? *)|(?:[fm] *to? *[fm] *))?(?:trans(?:guy|boy|girl|man|woman|gender|female|male))(?:(?: *(?:fe)?male(?: *to *(?:fe)?male)?)|(?:[fm] *to? *[fm]))?))|(?:[^a-z](?:[fm] *to? *[fm])[^a-z])|(?:(?:fe)?male *to *(?:fe)?male *))|(female)|(male)|(?:(?:sex|gender|(?:^|[^\.,a-z0-9])\d+)(?:^|(?:[^a-z0-9]*))(?:(f)|(m))[^a-z0-9])|(?:(?:^|[^a-z0-9])(?:(f)|(m))\d))'), # |(?:(penis|brother|boyfriend|bf)|(vagina|girlfriend))
                'key': ('Transgender', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male'),
                'acceptions' : dict(), # MAKE SURE LOWER CASE
                'conflict': AUTH_CONFLICT},
                'race':
                {'re':re.compile(r"(?i:[^a-z0-9](?:(white|caucasi[ao]n|european|irish|jewish|english|british|australian)|((?:black|african)(?: american)?)|(hispanic|mexican|south american)|(asian?(?:[^a-z0-9]american)?|chin(?:ese|a)|korean?|japan(?:ese)?)|(middle eastern|arabian|iran(?:ian)?|afghan|iraqi?|palastin(?:e|ian))|(poly(?:nesian)?|pacific(?:[^a-z0-9]island(?:er)?)|hawaii(?:an)?)|(native american|cherokee|navajo|choctaw|sioux|apache)|(indian)|((?:(?:mixed|multi)(?:rac(?:e|(?:ial))|ethnic(?:ity)?))|mixed)|(unknown|race[^a-z0-9]i (?:don'?t|do not) know)|(?:(?:race|ethnicity)[^a-z0-9])(?:(w)|(b)|(a)|(h))(?:[^a-z0-9]))(?=[^a-z0-9]))"),
                'key': ('White','Black','Hispanic', 'Asian', 'Middle_Eastern', 'Polynesian', 'Native_American', 'Indian', 'Multiracial', 'Unknown', 'White', 'Black', 'Asian', 'Hispanic'),
                'acceptions' : {'white' : -1, 'black': -1, 'english':-2, 'chinese':-1, 'british':-1, 'poly':-1}, # MAKE SURE LOWER CASE
                'conflict': 'Multiracial'}
                }


# Replaces newline and tab characters with an escaped "\" followed by the corresponding letter (for visualization purposes)
def mod_new_lines(text):
    # ret_text = re.sub(r'(\r|\n|\t)', r'\\$1', text)
    ret_text = re.sub(r'\r', r'\\r', text)
    ret_text = re.sub(r'\n', r'\\n', ret_text)
    ret_text = re.sub(r'\t', r'\\t', ret_text)
    return ret_text

# Check if a comment was created by the "auto-moderator"
def is_auto_moderator(comment_dic):
    return comment_dic["author"] in AUTO_MOD_AUTHOR_NAMES

# Drops columns in cols_to_drop from orig_dict (in place)
def drop_cols(orig_dict, cols_to_drop):
    mod_dict = orig_dict
    for k in [k for k in mod_dict if k in cols_to_drop]:
        del mod_dict[k]
    return mod_dict

# Identifies which group the corrent flair belongs in using the regex in the groups dict
# Note that the group index/number is currently unused (only need to return the group name)
def flair_identifier(flair, groups=GROUPS):
    if flair is None or flair.strip() == "":
        return -2, NONE_GROUP # return -1, UNKNOWN_GROUP
    flair_updated = flair.strip()
    # for i, (name, re_base) in enumerate(groups.items()):
    for i, (name, re_base) in enumerate(groups):
        if re_base.search(flair_updated) is not None:
            return i, name
    # return -1, UNKNOWN_GROUP
    return -1, ""


# Functions for gender and race
def areas_of_interest(text, layers=AUTH_LAYERS, distance=AUTH_DISTANCE_BUFFER, verbose=False):
    true_dist = int(distance) if distance >= 1 else int(len(text) * distance)
    if verbose:
        print(true_dist)

    locations = []
    for (layer, weight) in layers:
        locations.extend([(res.span(), weight) for res in layer.finditer(text)]) #:# results:
    
    groups = []
    locations.sort(key=lambda x: x[0][0])
    for (loc, loc_weight) in locations:
        window = (max(0, loc[0] - true_dist), min(len(text), loc[1] + true_dist))
        in_group = False
        for i, (g, g_weight) in enumerate(groups):
            if window[0] <= g[1]:
                groups[i] = ((min(g[0], window[0]), max(g[1], window[1])), g_weight + loc_weight)
                in_group = True
        if not in_group:
            groups.append(((window), loc_weight))
    groups.sort(key=lambda x: x[1], reverse=True)

    if verbose:
        print(locations)
        print(groups)
    return groups

def tuple_to_val(tuple_list, key, weight, acceptions, conflict=AUTH_CONFLICT, unspecified=AUTH_UNSPECIFIED, choose_first=True, verbose=False):
    def calc_vote(t, weight, acceptions, unspecified=AUTH_UNSPECIFIED):
        ind = min([i for i,key in enumerate(t) if key is not None and len(key) > 0])
        val = t[ind].lower()
        ret_weight = weight + acceptions[val] if val is not None and len(val) > 0 and val in acceptions else weight
        vote = key[ind] if ret_weight >= 0 else unspecified
        return (vote, ret_weight)
        
    if verbose:
        print(tuple_list)
    if tuple_list is None or isinstance(tuple_list, float):
        return AUTH_NONE
    elif len(tuple_list) == 0:
        return AUTH_UNSPECIFIED
    else:
        votes = [calc_vote(t, weight, acceptions, unspecified) for t in tuple_list]
        votes.sort(key=lambda x:x[1], reverse=True)
        first = votes[0][0]
        if len(votes) > 1 and not choose_first:
            for (v,w) in votes:
                if first == unspecified and w >= 0:
                    first = v
                elif v != first and v != unspecified and w >= 0:
                    return conflict
        return first

def find_groups(text, interest_dict=AUTH_INTERESTS, use_focus_groups=True, verbose=False):
    if not isinstance(text, str) or len(text) == 0 or BAD_TEXT_POSTS.fullmatch(text) is not None:
        return [AUTH_UNSPECIFIED for _ in interest_dict]
    groups = {k:[] for k in interest_dict}
    if use_focus_groups:
        focus_groups = areas_of_interest(text, verbose=verbose)
        for (k, val) in interest_dict.items():
            for (f_g, weight) in focus_groups:
                groups[k].append((tuple_to_val(val['re'].findall(text[f_g[0]:f_g[1]]), key=val['key'], weight=weight, acceptions=val['acceptions'], conflict=val['conflict'], choose_first=AUTH_CHOOSE_FIRST), weight))
    for (k, val) in interest_dict.items():
        groups[k].append((tuple_to_val(val['re'].findall(text), key=val['key'], weight=0, acceptions=val['acceptions'], conflict=val['conflict'], choose_first=AUTH_CHOOSE_FIRST, verbose=verbose), 0))
    if verbose:
        print(groups)
    for k, vals in groups.items():
        if vals is None or len(vals) == 0:
            groups[k] = AUTH_UNSPECIFIED
        else:
            decision = vals[0][0]
            decision_weight = vals[0][1]
            conflicts = []
            for (v, weight) in vals:
                if v != AUTH_UNSPECIFIED:
                    if (decision == AUTH_UNSPECIFIED) or (AUTH_SELECT_BY_WEIGHT and weight > decision_weight and v != AUTH_UNSPECIFIED):
                        decision = v
                        decision_weight = weight
                    elif ((not AUTH_SELECT_BY_WEIGHT) or (AUTH_SELECT_BY_WEIGHT and decision_weight == weight)) and (decision != v or v == AUTH_CONFLICT):
                        if len(conflicts) == 0:
                            conflicts.append((decision, decision_weight)) 
                        conflicts.append((v, weight))

            groups[k] = (decision, conflicts)
    if verbose:
        print(groups)
    return [v[0] for v in groups.values()]


# Process a comment and its nested/sub-comments returning a list of modified comment dicts, the comment count, minimum time to response and counts of comments from each flair group
def process_comments(comments_dict, post_id, parent_id, other_post_columns, flair_groups=GROUPS, parent_subreddit="askdocs"):
    comments_count = 0
    min_comment_time = None

    flair_counts = {k: 0 for k in {n for (n,_) in flair_groups}}
    ret_list = []

    for comment_id, comment in comments_dict.items():
        if not is_auto_moderator(comment):
            if min_comment_time is None:
                min_comment_time = comment[CC_TIME]
            else:
                min_comment_time = min(comment[CC_TIME], min_comment_time)

            comments_count += 1

            sub_comments_list, sub_cnt, min_sub_time, sub_flair_counts = [], 0, None, None
            # Process nested (sub) comments
            if CC_NESTED_COMMENTS in comment:
                sub_comments_list, sub_cnt, min_sub_time, sub_flair_counts = process_comments(comment[CC_NESTED_COMMENTS], post_id, comment_id, other_post_columns, flair_groups, parent_subreddit) # , comments_list #  , _
                comments_count += sub_cnt
                for s_key, s_val in sub_flair_counts.items():
                    flair_counts[s_key] += s_val
            else:
                # sub_flair_counts = {k: 0 for k in flair_groups}
                sub_flair_counts = {k: 0 for k in {n for (n,_) in flair_groups}}

            # For Testing purposes
            if SHOULD_MOD_NEWLINES:
                comment[CC_BODY] = mod_new_lines(comment[CC_BODY])

            comment_updated = dict(other_post_columns) # Making a copy\
            # Adding new key/value pairs to comment dict
            comment_updated[CC_POST_ID] = post_id
            comment_updated[CC_PARENT_ID] = parent_id
            comment_updated[CC_SUBREDDIT] = parent_subreddit
            if CC_COMMENT_ID not in comment:
                comment_updated[CC_COMMENT_ID] = comment_id
            comment_updated[CC_NESTED_COUNT] = sub_cnt
            comment_updated[CC_BODY_LEN] = len(comment[CC_BODY])
            comment_updated[CC_TIME_TO_NESTED] = min_sub_time - comment[CC_TIME] if min_sub_time is not None else NONE_VALUE
            comment_updated[CC_LINK_COUNT] = len(re.findall(HYPERLINK_SEARCH, comment[CC_BODY]))
            # add counts of flairs of subcomments when applicable
            if sub_flair_counts is not None:
                comment_updated.update(sub_flair_counts)
            
            _, group_name = flair_identifier(comment[CC_AUTHOR_FLAIR], flair_groups)
            comment_updated[CC_AUTHOR_FLAIR_GROUP] = group_name
            flair_counts[group_name] += 1

            comment = drop_cols(comment, CC_TO_DROP)

            # raise an error if a key about to be added already existed in the comment dict
            if RAISE_COMMENT_OVERWRITE:
                for k in comment_updated.keys():
                    if k in comment:
                        raise Exception(f"{k} key already exists in the comment.  Concider renaiming since this will override the original value (original= {k}:{comment[k]}; updated={k}:{comment_updated[k]})")

            comment_updated.update(comment)
            
            # Add processed comments (and nested comments) to the return list
            ret_list.append(comment_updated)
            ret_list.extend(sub_comments_list)
    return ret_list, comments_count, min_comment_time, flair_counts

def process_batch(batched_list, dest_path, is_first_batch=False):
    df_batch = pd.DataFrame(batched_list)
    if is_first_batch:
        df_batch.to_csv(dest_path, index=False)
    else:
        df_batch.to_csv(dest_path, mode='a', index=False, header=False)
    return []


# Read a json file and process a post and its comments, writing the processed data to destination files
def read_in_json(source_directory, destination_directory, flair_groups=GROUPS, author_interests=AUTH_INTERESTS):
    post_list = []
    comments_list = []

    post_dest = os.path.join(destination_directory, POST_NAME)
    comments_dest = os.path.join(destination_directory, COMMENT_NAME)

    file_list = [os.path.join(root, name) for root, _, file in os.walk(source_directory) for name in file if name.endswith(ACCEPTED_FILE_TYPES)]

    pbar = tqdm(file_list, disable=SHOULD_HIDE_PROGRESS_BAR)
    batch_cnt = 0
    auth_interst_names = [name for name in author_interests]
    for i, name in enumerate(pbar):
        pbar.set_description(name)

        post = json.load(open(name))

        other_post_cols = {key:val for key, val in post.items() if key in CC_OTHERS_FROM_POST}

        if "subreddit" in post:
            subreddit_name = post['subreddit']
        else:
            subreddit_name = "askdocs"

        parent_val = post[PC_POST_ID] if SET_POST_AS_PARENT else NONE_VALUE
        post_comments, count, min_sub_time, flair_counts = process_comments(post[PC_COMMENTS], post[PC_POST_ID], parent_val, other_post_cols, flair_groups, subreddit_name) #, comments_list) # , earliest_time
        
        comments_list.extend(post_comments)

        post_updated = drop_cols(post, PC_TO_DROP)

        auth_interest_title = find_groups(post_updated[PC_TITLE])
        auth_interest_body = find_groups(post_updated[PC_BODY])
        auth_interest_results = [i if i != AUTH_UNSPECIFIED else j for (i,j) in zip(auth_interest_title, auth_interest_body)]

        # For testing purposes
        if SHOULD_MOD_NEWLINES:
            post_updated[PC_BODY] = mod_new_lines(post_updated[PC_BODY])

        # Add new key/value pairs to dict
        post_updated[PC_SUBREDDIT] = subreddit_name
        post_updated[PC_BODY_LEN] = len(post_updated[PC_BODY])
        post_updated[PC_TIME_TO_COMMENT] = min_sub_time - post_updated[PC_TIME] if min_sub_time is not None else NONE_VALUE
        post_updated[PC_COMMENT_COUNT] = count
        post_updated[PC_LINK_COUNT] = len(re.findall(HYPERLINK_SEARCH, post[PC_BODY]))
        if flair_counts is not None:
            post_updated.update(flair_counts)
        
        _, group_name = flair_identifier(post[PC_AUTHOR_FLAIR], flair_groups) # flair_num
        post_updated[PC_AUTHOR_FLAIR_GROUP] = group_name

        for (k,v) in zip(auth_interst_names, auth_interest_results):
            post_updated[k] = v

        post_list.append(post_updated)

        if SHOULD_BATCH_SAVE and 0 < i and i % BATCH_SIZE == 0:
            if PRINT_SAVE_COUNTS:
                print(f"  Saving {len(post_list)} posts and {len(comments_list)} comments")
            post_list = process_batch(post_list, post_dest, batch_cnt == 0)
            comments_list = process_batch(comments_list, comments_dest, batch_cnt == 0)
            batch_cnt += 1

    if PRINT_SAVE_COUNTS:
        print(f"  Saving {len(post_list)} posts and {len(comments_list)} comments")
    if len(post_list) > 0:
        post_list = process_batch(post_list, post_dest, batch_cnt == 0)
    if len(comments_list) > 0:
        comments_list = process_batch(comments_list, comments_dest, batch_cnt == 0)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Improper usage:\n  To parse Json files: py ./csv_builder.py ./directory/of/jsons ./directory/to/save/csvs_to')
    source_directory = sys.argv[1]
    destination_directory = sys.argv[2] if len(sys.argv) > 2 else "./" 

    if RUN_PARSER:
        read_in_json(source_directory, destination_directory)

    if TEST_READ:
        df = pd.read_csv(os.path.join(destination_directory, COMMENT_NAME))

        print(df.head(10))
        for ind, val in pd.DataFrame(df[CC_AUTHOR_FLAIR].value_counts()).reset_index().to_numpy():
            print(ind, val)

