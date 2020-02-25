package cpww;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.process.Morphology;

import java.io.*;
import java.util.*;

public class Util {
    public static int sum(List<Integer> abc){
        int total = 0;
        for (Integer integer : abc) total += integer;
        return total;
    }

    static String min_characters(List<String> abc){
        String ans = abc.get(0);
        int total = ans.length();
        for (int i = 1; i < abc.size(); i++){
            if (abc.get(i).length() < total) {
                total = abc.get(i).length();
                ans = abc.get(i);
            }
        }
        return ans;
    }

    static List<Map.Entry<MetaPattern, Integer>> sortDecreasing(Map<MetaPattern, Integer> map) {
        List<Map.Entry<MetaPattern, Integer>> l = new ArrayList<>(map.entrySet());
        // Arrange patterns in decreasing order of frequency
        l.sort((o1, o2) -> o2.getValue().compareTo(o1.getValue()));
        return l;
    }

    static List<Integer> check_subsequence(List<SubSentWords> main_sequence, String mainPattern, String[] nerTypes){
        String[] pattern = mainPattern.split(" ");
        int m = main_sequence.size(), n = pattern.length;
        List<Integer> ans = new ArrayList<>();
        List<Integer> storingIndex = new ArrayList<>();
        List<String> patternTree = new ArrayList<>();
        boolean solutionFound;
        int startingIndex = 0;
        while (true) {
            int i = 0;
            solutionFound = true;
            for (int j = startingIndex ; j < m && i < n; j++) {
                if (main_sequence.get(j).getLemma().equals(pattern[i])) {
                    storingIndex.add(j);
                    patternTree.add(main_sequence.get(j).getTrimmedEncoding());
                    for (String ner : nerTypes) {
                        if (main_sequence.get(j).getLemma().contains(ner)) {
                            ans.add(j);
                        }
                    }
                    i++;
                }
            }
            if (i == n) {
                String root = min_characters(patternTree);
                for (String t : patternTree) {
                    if (!patternTree.contains(t.substring(0, t.length() - 1)) && !t.equals(root)) {
                        solutionFound = false;
                        startingIndex = storingIndex.get(0) + 1;
                        storingIndex.clear();
                        patternTree.clear();
                        ans.clear();
                        break;
                    }
                }
                if (solutionFound) {
                    return ans;
                }
            } else {
                return null;
            }
        }
    }

    static List<Map.Entry<SubSentWords, List<SubSentWords>>> sort_leafToTop(SentenceProcessor sentence){
        List<Map.Entry<SubSentWords, List<SubSentWords>>> l = new ArrayList<>(sentence.getSentenceBreakdown().entrySet());
        // Arrange sub-roots from leaf to root of sentence
        l.sort((o1, o2) -> {
            Integer i2 = o2.getKey().getEncoding().length();
            Integer i1 = o1.getKey().getEncoding().length();
            return i2.compareTo(i1);
        });
        return l;
    }

    static String returnLowercaseLemma(IndexedWord word) {
        if (word.lemma() == null) word.setLemma(Morphology.lemmaStatic(word.value(), word.tag()));
        return word.lemma().toLowerCase();
    }

    static boolean containsEntity(String subRoot, String[] nerTypes) {
        for (String ner : nerTypes) {
            if (subRoot.contains(ner)) {
                return true;
            }
        }
        return false;
    }

    static boolean containsEncoding(List<SubSentWords> list, String encoding) {
        for (SubSentWords w : list) {
            if (w.getEncoding().equals(encoding)) {
                return true;
            }
        }
        return false;
    }

    static SubSentWords returnSubSentWord(List<SubSentWords> list, String encoding) {
        for (SubSentWords w : list) {
            if (w.getTrimmedEncoding().equals(encoding)) {
                return w;
            }
        }
        return null;
    }

    static void updateMapCount(Map<String, List<Integer>> map, String token, Integer value) {
        List<Integer> temp = map.getOrDefault(token, new ArrayList<>());
        temp.add(value);
        map.put(token, temp);
    }

    static void writePatternsToFile(BufferedWriter bw, Map<String, List<MetaPattern>> patternList) throws IOException {
        for (String metaPattern : patternList.keySet()) {
            int freq = frequencySumHelper(patternList.get(metaPattern));
            bw.write(metaPattern + "->" + freq + "\n");
        }
        bw.close();
    }

    static int frequencySumHelper(List<MetaPattern> patterns) {
        int total = 0;
        for (MetaPattern p : patterns) total += p.getFrequency();
        return total;
    }

    static int maxNerCount(List<MetaPattern> patterns) {
        if (patterns.isEmpty()) return 0;
        int max = Integer.MIN_VALUE;
        for (MetaPattern p : patterns) {
            int temp = p.getNerCount();
            if (temp > max) max = temp;
        }
        return max;
    }

    static void checkAndSetParameter(Properties prop, String key, String value) {
        if (!prop.containsKey(key) || prop.getProperty(key).equals("")) {
            prop.setProperty(key, value);
        }
    }

    static void setOptionalParameterDefaults(Properties prop) {
        checkAndSetParameter (prop, "inputFolder", prop.getProperty("data_name") + "/");
        checkAndSetParameter (prop, "outputFolder", prop.getProperty("data_name") + "_Output/");
        checkAndSetParameter (prop, "load_sentenceBreakdownData", "false");
        checkAndSetParameter (prop, "load_metapatternData", "false");
        checkAndSetParameter (prop, "minimumSupport", "5");
        checkAndSetParameter (prop, "maxLength", "15");
        checkAndSetParameter (prop, "stopWordsFile", "stopWords.txt");
        checkAndSetParameter (prop, "nerTypes", prop.getProperty("inputFolder") + "entityTypes.txt");
    }

    static List<String> readList(FileReader fr) throws IOException {
        BufferedReader br = new BufferedReader(fr);
        List<String> ans = new ArrayList<>();
        String line = br.readLine();
        while (line!= null) {
            ans.add(line);
            line = br.readLine();
        }
        return ans;
    }
}
