-- Load dataset
accounts = LOAD '/assignment1/accounts.csv'
    USING PigStorage(',')
    AS (id:int, name:chararray, jobtitle:chararray, region:int, hobby:chararray);

follows = LOAD '/assignment1/follows.csv'
    USING PigStorage(',')
    AS (colrel:int, id1:int, id2:int, date:int, desc:chararray);

-- Get follower data
follower_join = JOIN follows BY id1, accounts BY id;

-- simplify follower data
follower_data = FOREACH follower_join GENERATE
    follows::id1 AS id1,
    follows::id2 AS id2,
    accounts::region AS region1;

-- Get followed data
followed_join = JOIN follower_data BY id2, accounts BY id;

-- simplify followed data
full_data = FOREACH followed_join GENERATE
    follower_data::id1 AS id1,
    follower_data::id2 AS id2,
    follower_data::region1 AS region1,
    accounts::region AS region2;

-- Get same region data
same_region = FILTER full_data BY
    region1 == region2;

-- Get reverse data, switch id1 and id2
reverse = FOREACH follows GENERATE id2 AS id1, id1 AS id2;

-- Get non reciprocal data
non_reciprocal = JOIN same_region BY (id1, id2)
    LEFT OUTER, reverse BY (id1, id2);

-- Get not followed back data by checking if reverse::id1 is null
not_followed_back = FILTER non_reciprocal
    BY reverse::id1 IS NULL;

-- Get result ids
result_ids = DISTINCT (FOREACH not_followed_back GENERATE same_region::id1);

-- Final join
final_join = JOIN result_ids BY id1, accounts BY id;

-- Get result
result = FOREACH final_join GENERATE
    accounts::id,
    accounts::name;

STORE result INTO '/assignment2/taskH'
    USING PigStorage(',');