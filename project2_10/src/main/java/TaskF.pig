-- Load dataset
accounts = LOAD '/assignment1/accounts.csv'
    USING PigStorage(',')
    AS (id:int, name:chararray, jobtitle:chararray, region:int, hobby:chararray);

follows = LOAD '/assignment1/follows.csv'
    USING PigStorage(',')
    AS (colrel:int, id1:int, id2:int, date:int, desc:chararray);

-- Group by id2
group_follows = GROUP follows BY id2;

-- Count followers
follow_count = FOREACH group_follows GENERATE
    group AS id,
    COUNT(follows) AS num_followers;

-- Join with accounts
joined = JOIN accounts BY id LEFT OUTER, follow_count BY id;

-- Get all counts
all_counts = FOREACH joined GENERATE
    accounts::id AS id,
    (follow_count::num_followers IS NOT NULL ? follow_count::num_followers : 0)
        AS num_followers;

-- Group all
group_all = GROUP all_counts ALL;

-- Calculate average
avg_followers = FOREACH group_all GENERATE
    AVG(all_counts.num_followers) AS avg_num;

-- Filter by average
popular_accounts = FILTER all_counts BY 
    num_followers > avg_followers.avg_num;

-- Store the result
STORE popular_accounts INTO '/assignment2/taskF'
    USING PigStorage(',');