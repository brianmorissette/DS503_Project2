import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;


// !!IMPORTANT!!
// For purpouses of this project the "ActionTime" column (random number 1-1,000,000) will
// be interpreted as "minutes since action happend" ie. 60 = action happend 1 hour ago
// So effectively the dataset covers 2 years worth of actions

public class TaskG {

    // Mapper: Extract user and hashtags
    public static class TaskGMapper extends Mapper<Object, Text, IntWritable, Text> {

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
        
        private IntWritable id = new IntWritable();
        private IntWritable time = new IntWritable();

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            
            String line = value.toString();
            String[] parts = line.split(",");

            // Skip header
            if (parts[0].equals("ActionId"))
                return;
            
            time.set(Integer.parseInt(parts[4]));
            id.set(Integer.parseInt(parts[1]));

            //Only write to the file if the action happened in the last 90 days (129,600 minutes)
            if(time.get() < 129600){
                context.write(id, new Text(users.get(id.get())));
            }
            
        }
    }

    public static void main(String[] args) throws Exception {

        String outputPath = args[0];

        // First Job: Count hashtags
        Configuration conf1 = new Configuration();
        Job job1 = Job.getInstance(conf1, "TaskG");

        job1.setJarByClass(TaskG.class);
        job1.setMapperClass(TaskGMapper.class);

        job1.setNumReduceTasks(0);

        job1.setMapOutputKeyClass(IntWritable.class);
        job1.setMapOutputValueClass(Text.class);

        job1.setOutputKeyClass(IntWritable.class);
        job1.setOutputValueClass(Text.class);

        // We found this function to optimize this to be a map only job
        // all user ID's and nicnames are cached in each mapper's memory
        // so we don't need to join with the accounts file.
        job1.addCacheFile(new Path("/assignment1/accounts.csv").toUri());


        FileInputFormat.addInputPath(job1, new Path("/assignment1/activity.csv"));
        FileOutputFormat.setOutputPath(job1, new Path(outputPath));

        System.exit(job1.waitForCompletion(true) ? 0 : 1);
    }
}
