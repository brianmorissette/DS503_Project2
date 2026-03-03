import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import java.io.IOException;
import org.apache.hadoop.fs.FileSystem;


import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class TaskF {

    //Mapper to get avg followers for all accounts
    public static class Sum_Followers_Mapper extends Mapper<Object, Text, IntWritable, IntWritable> {
        

        private IntWritable accountID = new IntWritable();
        private static final IntWritable one = new IntWritable(1);        

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            
            //take the record it and split it by column
            String line = value.toString();
            String[] attr = line.split(",");

            //Make sure we skip the header
                if(attr[0].equals("Colrel")) return;

                //set the key to the id of the second ID in the follow
                accountID.set(Integer.parseInt(attr[2]));

                //This is where we write the key value pair
                context.write(accountID, one);
        }
    }
    
    //Reducer to get avg followers for all accounts
    public static class Sum_Followers_Reducer extends Reducer<IntWritable,IntWritable,IntWritable,  IntWritable> {
        
        //be using an enum and writing to context we can skip having to use an "average" map reduce job
        enum Stats{
            totalFollowers,
            users;
        }

        public void reduce(IntWritable key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            

            //track the count for each Follower
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }   
            
            //write the key value pair
            context.write(key, new IntWritable(sum));

            //increment the counters
            context.getCounter(Stats.users).increment(1);
            context.getCounter(Stats.totalFollowers).increment(sum);
        }
    }

    //Mapper class for Task F
    public static class TASKFMapper extends Mapper<Object, Text, IntWritable, IntWritable> {

        private double avg;

        @Override
        protected void setup(Context context) {
            avg = context.getConfiguration().getDouble("avg.follows", 0.0);

        }

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            
            
            String line = value.toString().trim();


            if (line.isEmpty()) return;

            //Get the average from the previous job
            String[] parts = line.split("\\s+");

            //Get the user and total followers
            IntWritable user = new IntWritable(Integer.parseInt(parts[0]));
            IntWritable total = new IntWritable(Integer.parseInt(parts[1]));

            //Only write if the total followers is greater than the average
            if (total.get() > avg) {
                context.write(user, total);

            }
        }
        
    }
    
    //Main method
    public static void main(String[] args) throws Exception {


        // First Job sum followers for each account
        //Create a new job
        Configuration conf = new Configuration();
        Job job1 = Job.getInstance(conf, "Task F");
        
        //Set the job class
        job1.setJarByClass(TaskF.class);
        
        //Set the mapper class
        job1.setMapperClass(Sum_Followers_Mapper.class);
        
        //Set the reducer class
        job1.setReducerClass(Sum_Followers_Reducer.class);

        //Set the output key and value classes
        //This is where you set the data types for the key and value output
        job1.setOutputKeyClass(IntWritable.class);
        job1.setOutputValueClass(IntWritable.class);
        
        //Set the input and output paths
        //This is where you set the input and output paths
        FileInputFormat.addInputPath(job1, new Path("/assignment1/follows.csv"));
        FileOutputFormat.setOutputPath(job1, new Path("/tmp"));

        //Wait for job 1 to complete    
        job1.waitForCompletion(true);

        //Calculate the average in between jobs
        long totalFollows = job1.getCounters()
        .findCounter(Sum_Followers_Reducer.Stats.totalFollowers)
        .getValue();

        long userCount = job1.getCounters()
        .findCounter(Sum_Followers_Reducer.Stats.users)
        .getValue();

        double avg = (double) totalFollows / userCount;

        System.out.println("Average followers: " + avg);




        //Begin Job 2
        Configuration conf2 = new Configuration();
        conf2.setDouble("avg.follows", avg);
        Job job2 = Job.getInstance(conf2, "Task F");    

        //Set the job class
        job2.setJarByClass(TaskF.class);
        
        //Set the mapper class
        job2.setMapperClass(TASKFMapper.class);
        job2.setNumReduceTasks(0);

        job2.setMapOutputKeyClass(IntWritable.class);
        job2.setMapOutputValueClass(IntWritable.class);

        
        //Set the output key and value classes
        job2.setOutputKeyClass(IntWritable.class);
        job2.setOutputValueClass(IntWritable.class);

        //Get output of Job 1 and set output directory for Job 2
        FileInputFormat.addInputPath(job2, new Path("/tmp"));
        FileOutputFormat.setOutputPath(job2, new Path(args[0]));

        boolean success = job2.waitForCompletion(true);

        //Delete the temporary directory
        FileSystem fs1 = FileSystem.get(conf2);

        // path to the temp directory
        Path tempDir1 = new Path("/tmp");

        if (fs1.exists(tempDir1)) {
            fs1.delete(tempDir1, true); 
        }

        //Run the job
        System.exit(success ? 0 : 1);
    }
    
}
