
import com.fasterxml.jackson.core.type.TypeReference;
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
        props.setProperty("ner.model", "edu/stanford/nlp/models/ner/english.muc.7class.distsim.crf.ser.gz");
        props.setProperty("ner.applyFineGrained", "false");
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
        BufferedReader br = new BufferedReader(new FileReader("HK_news.txt"));
        BufferedWriter bw = new BufferedWriter(new FileWriter("HK_news_annotated.txt"));
        Map<String, String> map = new HashMap<>();
        int globalNER = 0;
        String line = br.readLine();
        StringBuilder sb;
        int count = 0;
        while (line != null) {
            String id = line.split("\t")[0];
            CoreDocument doc = new CoreDocument(line.split("\t")[1]);
            sb = new StringBuilder(id);
            // annotate the document
            pipeline.annotate(doc);
            // view results
            List<String> tokensAndNERTags = doc.tokens().stream().map(token -> token.word() + "<<->>" + token.ner().toUpperCase()).collect(
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
            line = br.readLine();
        }
        br.close();
        bw.close();
        String json = new ObjectMapper().writeValueAsString(map);
        try (FileWriter file = new FileWriter("HK_news_annotated" + "_dict.json")) {
            file.write(json);
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

}
