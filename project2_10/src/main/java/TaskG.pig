-- Load dataset
accounts = LOAD '/assignment1/accounts.csv'
    USING PigStorage(',')
    AS (id:int, name:chararray, jobtitle:chararray, region:int, hobby:chararray);

activity = LOAD '/assignment1/activity.csv'
    USING PigStorage(',')
    AS (actionid:int, userid:int, page:int, actiontype:chararray, actiontime:int);

-- Filter recent activity
recent_activity = FILTER activity BY actiontime <= 129600;

-- Get active users
active_users = DISTINCT (FOREACH recent_activity GENERATE userid);

-- Join with accounts
joined = JOIN accounts BY id LEFT OUTER, active_users BY userid;

-- Get outdated accounts by filtering out active users from the activity log
outdated_accounts = FILTER joined BY active_users::userid IS NULL;

result = FOREACH outdated_accounts GENERATE
    accounts::id,
    accounts::name;

-- Store the result
STORE result INTO '/assignment2/taskG'
    USING PigStorage(',');