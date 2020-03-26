package cpww;

import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.process.Morphology;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

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

    static List<Map.Entry<String, Integer>> sortDecreasing(Map<String, Integer> map) {
        List<Map.Entry<String, Integer>> l = new ArrayList<>(map.entrySet());
        // Arrange patterns in decreasing order of frequency
        l.sort((o1, o2) -> o2.getValue().compareTo(o1.getValue()));
        return l;
    }

    static List<Integer> check_subsequence(List<SubSentWords> main_sequence, boolean checkContinuity,
                                           String mainPattern, String[] nerTypes){
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
                    if (containsEntity(main_sequence.get(j).getLemma(), nerTypes)) {
                        ans.add(j);
                    }
                    i++;
                }
            }
            if (i == n) {
                if (checkContinuity) {
                    String root = min_characters(patternTree);
                    for (String t : patternTree) {
                        if (!patternTree.contains(t.substring(0, t.length() - 1)) && !t.equals(root)) {
                            solutionFound = false;
                            break;
                        }
                    }
                    if (solutionFound && noConjugateCheck(main_sequence, storingIndex)) {
                        return ans;
                    } else {
                        startingIndex = storingIndex.get(0) + 1;
                        storingIndex.clear();
                        patternTree.clear();
                        ans.clear();
                    }
                } else {
                    return ans;
                }
            } else {
                return null;
            }
        }
    }

    static List<SubSentWords> sort_leafToTop(SentenceProcessor sentence){
        List<SubSentWords> l = new ArrayList<>(sentence.getSentenceBreakdown().keySet());
        // Arrange sub-roots from leaf to root of sentence
        l.sort((o1, o2) -> {
            Integer i2 = o2.getTrimmedEncoding().length();
            Integer i1 = o1.getTrimmedEncoding().length();
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

    static int returnEncodeIndex(List<SubSentWords> encodeList, String encoding) {
        for (int i = 0; i < encodeList.size(); i++) {
            if (encodeList.get(i).getTrimmedEncoding().equals(encoding)) {
                return i;
            }
        }
        return -1;
    }

    static void updateMapCount(Map<String, List<Integer>> map, String token, Integer value) {
        List<Integer> temp = map.getOrDefault(token, new ArrayList<>());
        temp.add(value);
        map.put(token, temp);
    }

    static void writePatternsToFile(BufferedWriter bw, Map<String, List<MetaPattern>> patternList) throws IOException {
        Map<String, Integer> patternCount = new HashMap<>();
        patternList.forEach((key, val) -> patternCount.put(key, frequencySumHelper(val)));
        for (Map.Entry<String, Integer> metaPattern : sortDecreasing(patternCount)) {
            int freq = metaPattern.getValue();
            bw.write(metaPattern.getKey() + "->" + freq + "\n");
        }
        bw.close();
    }

    static int frequencySumHelper(List<MetaPattern> patterns) {
        int total = 0;
        for (MetaPattern p : patterns) total += p.getFrequency();
        return total;
    }

    static Integer maxNerCount(List<MetaPattern> patterns) {
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
        checkAndSetParameter (prop, "inputFolder", "Data/" + prop.getProperty("data_name") + "/");
        checkAndSetParameter (prop, "outputFolder", folderNameConsistency(prop.getProperty("inputFolder")) + "Output/");
        checkAndSetParameter (prop, "load_sentenceBreakdownData", "false");
        checkAndSetParameter (prop, "load_metapatternData", "false");
        checkAndSetParameter (prop, "minimumSupport", "5");
        checkAndSetParameter (prop, "maxLength", "15");
        checkAndSetParameter (prop, "stopWordsFile", "stopWords.txt");
        checkAndSetParameter (prop, "nerTypes", folderNameConsistency(prop.getProperty("inputFolder")) + "entityTypes.txt");
        checkAndSetParameter(prop, "noOfPushUps", "3");
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

    static Map<String, Integer> returnSortedPatternList(Map<String, List<MetaPattern>> patternList) {
        Map<String, Integer> ans = new LinkedHashMap<>();
        List<String> l = new ArrayList<>(patternList.keySet());
        l.sort((o1, o2) ->
                returnPatternWeight(o2, patternList.get(o2)).compareTo(returnPatternWeight(o1, patternList.get(o1))));
        l.forEach(s -> ans.put(s, maxNerCount(patternList.get(s))));
        return ans;
    }

    static String folderNameConsistency(String folderName) {
        String folder = folderName;
        if (folder.charAt(folder.length() - 1) != '/') {
            folder += "/";
        }
        return folder;
    }

    static String trimEncoding(String encoding) {
        return encoding.split("_")[0];
    }

    private static Double returnPatternWeight(String metaPattern, List<MetaPattern> patternList) {
        return (double) maxNerCount(patternList) + metaPattern.split(" ").length/20.0;
    }

    private static boolean noConjugateCheck(List<SubSentWords> mainSequence, List<Integer> storingIndex) {
        List<String> encoding = storingIndex.stream().map(s -> mainSequence.get(s).getEncoding()).collect(Collectors.toList());
        String parent = null;
        for (String e : encoding) {
            if (e.contains("conj")) {
                parent = e.split("_")[0];
                parent = parent.substring(0, parent.length() - 1);
                break;
            }
        }
        if (parent == null) return true;
        for (String e : encoding) {
            if (e.split("_")[0].equals(parent)) {
                return false;
            }
        }
        return true;
    }
}
