import com.fasterxml.jackson.databind.ObjectMapper;
import edu.stanford.nlp.pipeline.*;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

public class NERAnnotation {

    public static void main(String[] args) throws IOException {
        // set up pipeline properties
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner");
        props.setProperty("ner.model", "edu/stanford/nlp/models/ner/english.all.3class.distsim.crf.ser.gz");
        props.setProperty("ner.applyFineGrained", "true");
        // example customizations (these are commented out but you can uncomment them to see the results

        // disable fine grained ner
        // props.setProperty("ner.applyFineGrained", "false");

        // customize fine grained ner
        // props.setProperty("ner.fine.regexner.mapping", "example.rules");
        // props.setProperty("ner.fine.regexner.ignorecase", "true");

        // add additional rules, customize TokensRegexNER annotator
        // props.setProperty("ner.additional.regexner.mapping", "example.rules");
        // props.setProperty("ner.additional.regexner.ignorecase", "true");

        // add 2 additional rules files ; set the first one to be case-insensitive
        // props.setProperty("ner.additional.regexner.mapping", "ignorecase=true,example_one.rules;example_two.rules");

        // set document date to be a specific date (other options are explained in the document date section)
        // props.setProperty("ner.docdate.useFixedDate", "2019-01-01");

        // only run rules based NER
        // props.setProperty("ner.rulesOnly", "true");

        // only run statistical NER
        // props.setProperty("ner.statisticalOnly", "true");

        // set up pipeline
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
        // make an example document
        BufferedReader br = new BufferedReader(new FileReader("Data/nyt/nyt.txt"));
        BufferedWriter bw = new BufferedWriter(new FileWriter("Data/nyt/nyt3_annotated.txt"));
        BufferedWriter bw1 = new BufferedWriter(new FileWriter("Data/nyt/entityTypes.txt"));
        Set<String> entityTypes = new HashSet<>();
        Map<String, String> map = new HashMap<>();
        int globalNER = 0;
        String line = br.readLine();
        final boolean indexGiven = (line != null && line.split("\t").length != 1);
        StringBuilder sb;
        int count = 0;
        int lineNo = 0;
        while (line != null) {
            String id = indexGiven ? line.split("\t")[0] : String.valueOf(lineNo);
            String actualSent = indexGiven ? line.split("\t")[1] : line;
            if (actualSent.split(" ").length < 100) {
                CoreDocument doc = new CoreDocument(actualSent);
                sb = new StringBuilder(id);
                // annotate the document
                pipeline.annotate(doc);
                // view results
                List<String> tokensAndNERTags = doc.tokens().stream().map(token -> token.word() + "<<->>" + token.ner().toUpperCase().replaceAll("_", "")).collect(
                        Collectors.toList());
                String prev = "";
                List<String> sent = new ArrayList<>();
                List<String> subAns = new ArrayList<>();
                for (String s : tokensAndNERTags) {
                    String[] temp = s.split("<<->>");
                    if (!temp[1].equals(prev)) {
                        if (prev.equals("O")) {
                            sent.add(String.join(" ", subAns));
                        } else if (!prev.isEmpty()) {
                            map.put(prev + globalNER, String.join(" ", subAns));
                            sent.add(prev + globalNER++);
                        }
                        subAns.clear();
                    }
                    subAns.add(temp[0]);
                    prev = temp[1];
                    entityTypes.add(prev);
                }
                if (prev.equals("O")) {
                    sent.add(String.join(" ", subAns));
                } else {
                    map.put(prev + globalNER, String.join(" ", subAns));
                    sent.add(prev + globalNER++);
                }
                sb.append("\t").append(String.join(" ", sent)).append("\n");
                bw.write(sb.toString());
                count++;
                if (count % 100 == 0) System.out.println(sb.toString());
            }
            line = br.readLine();
            lineNo++;
        }
        br.close();
        bw.close();
        String json = new ObjectMapper().writeValueAsString(map);
        try (FileWriter file = new FileWriter("Data/nyt/nyt3" + "_dict.json")) {
            file.write(json);
        } catch (Exception e) {
            e.printStackTrace();
        }
        for (String entity : entityTypes) {
            bw1.write(entity + "\n");
        }
        bw1.close();

    }

}
