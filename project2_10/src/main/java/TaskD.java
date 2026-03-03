import java.io.IOException;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;


//Should have a combiner
public class TaskD {

    //Mapper class for Task D
    public static class TaskDMapper extends Mapper<Object, Text, IntWritable, IntWritable> {

        private IntWritable accountID = new IntWritable();
        private static final IntWritable one = new IntWritable(1);

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            
            //take the record it and split it by column
            String line = value.toString();
            String[] attr = line.split(",");

            if (attr.length < 3) return;
            if ("Colrel".equals(attr[0])) return;

            //set the key to the id of the second ID in the follow
            accountID.set(Integer.parseInt(attr[2]));

            //This is where we write the key value pair
            context.write(accountID, one);  
        }
    }

    //Reducer class for Task D
    public static class TaskDReducer extends Reducer<IntWritable,IntWritable,Text,IntWritable> {

        //HashMap to store user ID and nickname
        private Map<Integer, String> users = new HashMap<>();

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            //Get the cached file
            java.net.URI[] cacheFiles = context.getCacheFiles();

            if (cacheFiles != null && cacheFiles.length > 0) {

                // Extract the local filename from the URI
                String localFileName = new Path(cacheFiles[0].getPath()).getName(); 

                //Read the file
                try (BufferedReader reader = new BufferedReader(new FileReader(localFileName))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        String[] parts = line.split(",");

                        //Skip header
                        if (parts[0].equals("id"))
                            continue;
                        
                        int userId = Integer.parseInt(parts[0]);
                        String nickname = parts[1];
                        users.put(userId, nickname);
                    }
                }
            }
        }
        
        public void reduce(IntWritable key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            
            //Get the nickname from the users HashMap
            String nickname = users.get(key.get());

            //track the count for each Follower
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }   
            
            //write the key value pair
            context.write(new Text(nickname), new IntWritable(sum));
        }
    }

    // Combiner: same input/output types as mapper output; sums 1s locally
    public static class TaskDCombiner extends Reducer<IntWritable, IntWritable, IntWritable, IntWritable> {

        private IntWritable partialSum = new IntWritable();

        public void reduce(IntWritable key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }

            partialSum.set(sum);
            context.write(key, partialSum);
        }
    }

    //Main method
    public static void main(String[] args) throws Exception {

        //Create a new job
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Task D");
        
        //Set the job class
        job.setJarByClass(TaskD.class);
        
        //Set the mapper class
        job.setMapperClass(TaskDMapper.class);
        
        //Set the reducer class
        job.setReducerClass(TaskDReducer.class);

        //Set the combiner class
        job.setCombinerClass(TaskDCombiner.class);

        job.addCacheFile(new Path("/input/user/ds503/project1/accounts.csv").toUri());
        FileInputFormat.addInputPath(job, new Path("/input/user/ds503/project1/follows.csv"));

        // Map output
        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(IntWritable.class);

        // Final output
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        
        FileOutputFormat.setOutputPath(job, new Path(args[0]));
        
        //Run the job
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }   


}
