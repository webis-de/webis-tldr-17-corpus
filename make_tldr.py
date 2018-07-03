import sys
import argparse
import itertools
import csv
import re
from pyspark import SparkContext,SparkConf,SQLContext
from pyspark.sql import Row,SparkSession,HiveContext
from pyspark.sql.functions import col,size,explode,split
from pyspark.sql.types import StringType,IntegerType,ArrayType
from pyspark.sql.functions import udf, array, length

spark = SparkSession.builder.appName("webis-tldr-17-corpus-construction").getOrCreate()
sc = spark.sparkContext
print(spark)
print(sc)

parser = argparse.ArgumentParser(description="Construct summarization corpus from Reddit")
parser.add_argument('--input_comments', type=str, help="HDFS path to Reddit comments")
parser.add_argument('--input_submissions', type=str, help="HDFS path to Reddit submissions")
parser.add_argument('--output_comments', type=str, help="HDFS path to save processed comments")
parser.add_argument('--output_submissions', type=str, help="HDFS path to save processed submissions")

args = parser.parse_args()
input_comments = str(args.input_comments)
input_submissions = str(args.input_submissions)
output_comments = str(args.output_comments)
output_submissions = str(args.output_submissions)

'''
Read the submissions corpora and create a unified schema for from different years based on common fields. 
Finally, join them into a single dataframe
'''
submissions_df = spark.read.json(input_submissions)
submissions_df = submissions_df.select("author","id","selftext","title")
# Read comments corpus
comments_df = spark.read.json(input_comments)
comments_df = comments_df.select("author","id","body")
print("Intial number of comments: {}".format(comments_df.count()))
print("Intial number of submissions: {}".format(submissions_df.count()))
'''
Preprocessing pipeline consists of the following steps
1. Remove posts where body/selftext == '[deleted]' 
2. Remove posts by bots and deleted authors
3. Find posts containing any form of a tl.{0,3}dr pattern
4. Find posts containing only a single occurrence of such pattern
5. Find posts containing a valid pattern
6. Find the location of this pattern in a post and split the text into content-summary pair of appropriate lengths (content > summary)

'''

'''
Initialize global variables for markdown and list of bots
Sources:
[1] https://pypi.python.org/pypi/mistune
[2] https://www.reddit.com/r/autowikibot/wiki/redditbots
'''

global botlist
with open('./resources/botlist.csv','r') as bot_file:
    reader=csv.reader(bot_file)
    botlist=list(reader)
botlist = list(itertools.chain.from_iterable(botlist))

# Check for presence of a tldr pattern
def tl_dr(input):
    lower_text = str(input).lower()
    match = re.search(r'tl.{0,3}dr',str(lower_text))
    if match:
        return input
    else:
        return None

# Find all the matched tldr patterns
def get_all_tldr(input):
    lower = str(input).lower()
    pattern=re.compile(r'tl.{0,3}dr') 
    return pattern.findall(lower)

# Find location of the tldr pattern and split text to form <content, summary> pairs
def iter_tldr(text):
    lower_text = str(text).lower()
    patterns = re.findall(r'tl.{0,3}dr',lower_text)
    if len(patterns) > 0:
        match = patterns[-1]
        if match:
            index = lower_text.rfind(match)
            if index == 0 or index+len(match) == len(str(lower_text)):
                return None
            else:
                content = text[:index].strip()
                summary = text[index+len(match):].strip()
                if len(content.split()) > len(summary.split()):
                    return [content,summary]
                else:
                    return None
        else:
            return None
    else:
        return None

# Define all the UDFs to be applied on the dataframes
removeBotPosts = udf(lambda input:None if str(input) in botlist else input, StringType())
removeShortPosts = udf(lambda input:input if len(str(input).strip().split()) >= 100 else None,StringType())
findTldr = udf(tl_dr,StringType())
getAllTldrPatterns = udf(get_all_tldr,ArrayType(StringType()))
removeInvalidTldrPatterns=udf(lambda input:input if str(input).lower() in ["tl dr","tl;dr","tldr","tl:dr","tl/dr","tl; dr","tl,dr","tl, dr","tl-dr","tl'dr","tl: dr","tl.dr","tl ; dr","tl_dr","tldr;dr","tl ;dr","tl\"dr","tl/ dr","tld:dr","tl;;dr","tltl;dr","tl~dr","tl / dr","tl :dr","tl - dr","tl\\dr","tl. dr","tl:;dr","tl|dr","tl;sdr","tll;dr","tl : dr","tld;dr"] else None, StringType())
removeMultipleTldrs = udf(lambda input:input if len(input)==1 else None,ArrayType(StringType()))
getContentSummaryPair = udf(iter_tldr,ArrayType(StringType()))
getContent = udf(lambda input:input[0])
getSummary = udf(lambda input:input[1])

#remove posts where text == '[deleted]'
comments_df = comments_df.where(comments_df['body']!='[deleted]')
submissions_df = submissions_df.where(submissions_df['selftext']!='[deleted]')
print("After removing '[deleted]' -> comments: {}".format(comments_df.count()))
print("After removing '[deleted]' -> submissions: {}".format(submissions_df.count()))

# remove bots
comments_df = comments_df.withColumn('author', removeBotPosts(comments_df.author))
submissions_df = submissions_df.withColumn('author', removeBotPosts(submissions_df.author))
comments_df = comments_df.filter(comments_df.author.isNotNull())
submissions_df = submissions_df.filter(submissions_df.author.isNotNull())
print("After removing bots -> comments: {}".format(comments_df.count()))
print("After removing bots -> submissions: {}".format(submissions_df.count()))

""" # remove short posts
comments_df = comments_df.withColumn('body', removeShortPosts(comments_df.body))
submissions_df = submissions_df.withColumn('selftext', removeShortPosts(submissions_df.selftext))
comments_df = comments_df.filter(comments_df.body.isNotNull())
submissions_df = submissions_df.filter(submissions_df.selftext.isNotNull())
print("After removing short posts -> comments: {}".format(comments_df.count()))
print("After removing short posts -> submissions: {}".format(submissions_df.count())) """

# check if tldr is present
comments_df = comments_df.withColumn('body', findTldr(comments_df.body))
submissions_df = submissions_df.withColumn('selftext', findTldr(submissions_df.selftext))
comments_df = comments_df.filter(comments_df.body.isNotNull())
submissions_df = submissions_df.filter(submissions_df.selftext.isNotNull())
print("After locating a tl;dr pattern -> comments: {}".format(comments_df.count()))
print("After locating a tl;dr pattern -> submissions: {}".format(submissions_df.count()))

# get all patterns
comments_df = comments_df.withColumn('matched_tldrs', getAllTldrPatterns(comments_df.body))
submissions_df = submissions_df.withColumn('matched_tldrs', getAllTldrPatterns(submissions_df.selftext))
print("After locating all existing patterns -> comments: {}".format(comments_df.count()))
print("After locating all existing patterns -> submissions: {}".format(submissions_df.count()))

# preserve posts with only single tldr pattern
comments_df = comments_df.withColumn('matched_tldrs', removeMultipleTldrs(comments_df.matched_tldrs))
submissions_df = submissions_df.withColumn('matched_tldrs', removeMultipleTldrs(submissions_df.matched_tldrs))
comments_df = comments_df.filter(comments_df.matched_tldrs.isNotNull())
submissions_df = submissions_df.filter(submissions_df.matched_tldrs.isNotNull())
print("After removing multiple patterns -> comments: {}".format(comments_df.count()))
print("After removing multiple patterns -> submissions: {}".format(submissions_df.count()))

# add a new column with the tldr_tag
comments_df = comments_df.select("*",explode(col("matched_tldrs")).alias("tldr_tag"))
submissions_df = submissions_df.select("*",explode(col("matched_tldrs")).alias("tldr_tag"))

# remove noisy, less frequent tldr patterns
comments_df = comments_df.withColumn('tldr_tag',removeInvalidTldrPatterns(comments_df.tldr_tag))
submissions_df = submissions_df.withColumn('tldr_tag',removeInvalidTldrPatterns(submissions_df.tldr_tag))
comments_df = comments_df.filter(comments_df.tldr_tag.isNotNull())
submissions_df = submissions_df.filter(submissions_df.tldr_tag.isNotNull())
print("After removing invalid tl;dr -> comments: {}".format(comments_df.count()))
print("After removing invalid tl;dr -> submissions: {}".format(submissions_df.count()))

# extract content-summary pair - tldr pattern occuring only in valid location in text
comments_df = comments_df.withColumn('pair', getContentSummaryPair(comments_df.body))
submissions_df = submissions_df.withColumn('pair', getContentSummaryPair(submissions_df.selftext))
comments_df = comments_df.filter(comments_df.pair.isNotNull())
submissions_df = submissions_df.filter(submissions_df.pair.isNotNull())
print("After constructing valid pairs -> comments: {}".format(comments_df.count()))
print("After constructing valid pairs -> submissions: {}".format(submissions_df.count()))

# Create individual columns for content and summary
comments_df = comments_df.withColumn('content',getContent(comments_df.pair))
comments_df = comments_df.withColumn('summary',getSummary(comments_df.pair))
submissions_df= submissions_df.withColumn('content',getContent(submissions_df.pair))
submissions_df= submissions_df.withColumn('summary',getSummary(submissions_df.pair))
print("After extracting content, summary -> comments: {}".format(comments_df.count()))
print("After extracting content, summary -> submissions: {}".format(submissions_df.count()))

# Save the results 
submissions_df.write.json(output_submissions)
print("Wrote final submissions to HDFS at {}".format(output_submissions))
comments_df.write.json(output_comments)
print("Wrote final comments to HDFS at {}".format(output_comments))








