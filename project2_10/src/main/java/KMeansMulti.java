import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class KMeansMulti {

    // Data bounds for random seeds
    private static final double W_MIN = 0.0, W_MAX = 10_000.0;
    private static final double X_MIN = 20_000.0, X_MAX = 1_000_000.0;
    private static final double Y_MIN = 0.0, Y_MAX = 500_000.0;
    private static final double Z_MIN = 0.0, Z_MAX = 50_000.0;

    public static class KMeansMapper extends Mapper<Object, Text, IntWritable, Text> {
        private List<double[]> centroids = new ArrayList<>();
        private IntWritable nearestCentroidId = new IntWritable();
        private Text pointText = new Text();

        // Load centroids from DistributedCache
        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            Path[] localFiles = DistributedCache.getLocalCacheFiles(context.getConfiguration());
            if (localFiles == null || localFiles.length == 0) {
                throw new IOException("No seeds file in DistributedCache");
            }
            Path seedsPath = localFiles[0];
            FileSystem localFs = FileSystem.getLocal(context.getConfiguration());
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(localFs.open(seedsPath)))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    line = line.trim();
                    if (line.isEmpty()) continue;
                    String[] parts = line.split(",");
                    centroids.add(new double[]{
                        Double.parseDouble(parts[0]), Double.parseDouble(parts[1]),
                        Double.parseDouble(parts[2]), Double.parseDouble(parts[3])
                    });
                }
            }
        }

        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();

            // Skip the header row
            if (line.startsWith("w,x,y,z") || line.trim().isEmpty()) {
                return;
            }

            String[] parts = line.split(",");
            double[] point = new double[]{
                Double.parseDouble(parts[0]), Double.parseDouble(parts[1]),
                Double.parseDouble(parts[2]), Double.parseDouble(parts[3])
            };

            int nearestId = -1;
            double minDistance = Double.MAX_VALUE;

            // Find nearest centroid using 4D Euclidean distance
            for (int i = 0; i < centroids.size(); i++) {
                double[] c = centroids.get(i);
                double distance = Math.sqrt(
                    Math.pow(c[0] - point[0], 2) + Math.pow(c[1] - point[1], 2) +
                    Math.pow(c[2] - point[2], 2) + Math.pow(c[3] - point[3], 2)
                );

                if (distance < minDistance) {
                    minDistance = distance;
                    nearestId = i;
                }
            }

            nearestCentroidId.set(nearestId);
            pointText.set(line);
            context.write(nearestCentroidId, pointText);
        }
    }

    public static class KMeansReducer extends Reducer<IntWritable, Text, IntWritable, Text> {
        private Text newCentroidText = new Text();

        // Average points by cluster to compute new centroid
        @Override
        public void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            double sumW = 0, sumX = 0, sumY = 0, sumZ = 0;
            int count = 0;

            for (Text val : values) {
                String[] parts = val.toString().split(",");
                sumW += Double.parseDouble(parts[0]);
                sumX += Double.parseDouble(parts[1]);
                sumY += Double.parseDouble(parts[2]);
                sumZ += Double.parseDouble(parts[3]);
                count++;
            }

            double newW = sumW / count;
            double newX = sumX / count;
            double newY = sumY / count;
            double newZ = sumZ / count;

            newCentroidText.set(newW + "," + newX + "," + newY + "," + newZ);
            context.write(key, newCentroidText);
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 4) {
            System.err.println("Usage: KMeansMulti <k> <R> <input path> <output path>");
            System.exit(-1);
        }

        int k, R;
        try {
            k = Integer.parseInt(args[0]);
            R = Integer.parseInt(args[1]);
            if (k <= 0 || R <= 0) {
                System.err.println("Error: k and R must be a positive integers");
                System.exit(-1);
                return;
            }
        } catch (NumberFormatException e) {
            System.err.println("Error: k and R must be a positive integers");
            System.exit(-1);
            return;
        }

        Configuration conf = new Configuration();
        String inputPath = args[2];
        String outputPath = args[3];

        // Generate k random seeds within data bounds (only once)
        Random random = new Random();
        File localSeedsFile = File.createTempFile("kmeans_seeds_", ".txt");
        try (PrintWriter writer = new PrintWriter(localSeedsFile)) {
            for (int i = 0; i < k; i++) {
                double w = W_MIN + random.nextDouble() * (W_MAX - W_MIN);
                double x = X_MIN + random.nextDouble() * (X_MAX - X_MIN);
                double y = Y_MIN + random.nextDouble() * (Y_MAX - Y_MIN);
                double z = Z_MIN + random.nextDouble() * (Z_MAX - Z_MIN);
                writer.println(w + "," + x + "," + y + "," + z);
            }
        }

        // Upload initial seeds to HDFS
        FileSystem fs = FileSystem.get(conf);
        Path hdfsSeedsPath = new Path("/tmp/kmeans_seeds_" + System.currentTimeMillis() + ".txt");
        fs.copyFromLocalFile(new Path(localSeedsFile.getAbsolutePath()), hdfsSeedsPath);
        localSeedsFile.delete();

        // Run R iterations; each iteration uses previous centroids as seeds
        boolean success = true;
        for (int iter = 0; iter < R; iter++) {
            String iterOutputPath = outputPath + "/iter_" + iter;

            Job job = Job.getInstance(conf, "4D K-Means Iteration " + iter);
            job.setJarByClass(KMeansMulti.class);
            job.setMapperClass(KMeansMapper.class);
            job.setReducerClass(KMeansReducer.class);

            job.setOutputKeyClass(IntWritable.class);
            job.setOutputValueClass(Text.class);

            job.addCacheFile(hdfsSeedsPath.toUri());

            FileInputFormat.addInputPath(job, new Path(inputPath));
            FileOutputFormat.setOutputPath(job, new Path(iterOutputPath));

            success = job.waitForCompletion(true);
            if (!success) {
                System.err.println("Job failed at iteration " + iter);
                System.exit(1);
            }

            // Prepare seeds for next iteration: read reducer output, write to local file, upload to HDFS
            if (iter < R - 1) {
                Path newSeedsPath = new Path("/tmp/kmeans_seeds_iter_" + (iter + 1) + ".txt");

                // Read all part-r-* files from reducer output (format: clusterID\tw,x,y,z)
                try (PrintWriter writer = new PrintWriter(new File(newSeedsPath.getName()))) {
                    org.apache.hadoop.fs.FileStatus[] outputFiles =
                        fs.listStatus(new Path(iterOutputPath),
                            path -> path.getName().startsWith("part-r-"));

                    for (org.apache.hadoop.fs.FileStatus fileStatus : outputFiles) {
                        try (BufferedReader reader = new BufferedReader(
                                new InputStreamReader(fs.open(fileStatus.getPath())))) {
                            String line;
                            while ((line = reader.readLine()) != null) {
                                String centroid = line.split("\t")[1];
                                writer.println(centroid);
                            }
                        }
                    }
                }

                fs.copyFromLocalFile(new Path(newSeedsPath.getName()), newSeedsPath);
                new File(newSeedsPath.getName()).delete();
                hdfsSeedsPath = newSeedsPath;
            }
        }
        
        System.exit(success ? 0 : 1);
    }
}

