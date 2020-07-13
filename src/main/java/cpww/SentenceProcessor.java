package cpww;

import java.io.Serializable;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.LinkedHashMap;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.TreeSet;
import java.util.Queue;
import java.util.Collections;

import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.BasicDependenciesAnnotation;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.semgraph.SemanticGraph;

import static cpww.utils.Util.*;

public class SentenceProcessor implements Serializable {
    private String sentence;
    private String sentID;
    private Map<SubSentWords, List<SubSentWords>> sentenceBreakdown = new LinkedHashMap<>();
    private Map<String, SubSentWords> reverseEncoding = new HashMap<>();
    private Map<String, SubSentWords> replaceSurfaceName = new HashMap<>();
    private Map<SubSentWords, List<SubSentWords>> pushedUpSentences = new HashMap<>();

    public SentenceProcessor(String text, String sentID) {
        setSentenceAndID(text, sentID);
    }

    public void processSentence(StanfordCoreNLP pipeline, Map<String, String> entityDict, String[] nerTypes) {
        if (this.sentence == null) return;
        SemanticGraph semanticGraph = generateSemanticGraph(this.sentence.trim(), pipeline); //.replace("-", "_")
//        for (IndexedWord iw: semanticGraph.vertexListSorted()) {
//            if (iw.word().contains("_")) iw.setWord(iw.word().replace("_", "-"));
//        }
        Map<IndexedWord, String> encodeTree = encodeTree(semanticGraph, semanticGraph.getFirstRoot(), "A_root", new HashMap<>());
        generateReverseWordEncoding(encodeTree, entityDict, nerTypes);
        Map<IndexedWord, TreeSet<IndexedWord>> subTree = splitNoun(semanticGraph.getFirstRoot(), new HashMap<>(),  semanticGraph, nerTypes);
        sentenceBreakdown(encodeTree, subTree);
    }

    private SemanticGraph generateSemanticGraph(String text, StanfordCoreNLP pipeline) {
        Annotation ann = new Annotation(text);
        pipeline.annotate(ann);
        CoreMap sentence = ann.get(SentencesAnnotation.class).get(0);
        return sentence.get(BasicDependenciesAnnotation.class);
    }

    private Map<IndexedWord, TreeSet<IndexedWord>> splitNoun(IndexedWord root, Map<IndexedWord,
            TreeSet<IndexedWord>> subTree, SemanticGraph semanticGraph, String[] nerTypes) {
        int index = root.index();
        boolean rootIsVerb = root.tag().charAt(0) == 'V';
        subTree.put(root, subTree.getOrDefault(root, new TreeSet<>()));
        Queue<IndexedWord> search = new LinkedList<>();
        List<IndexedWord> potentialDeletions = null;
        search.offer(root);
        while (!search.isEmpty()) {
            IndexedWord node = search.poll();
            if (node.equals(root) && subTree.get(root).contains(node)) continue;
            boolean nodeIsVerb = node.tag().charAt(0) == 'V';
            TreeSet<IndexedWord> temp = subTree.get(root);
            temp.add(node);
            subTree.put(root, temp);
            int newIndex = node.index();
            boolean rootCheck = semanticGraph.getRoots().contains(node);
            List<IndexedWord> children;
            if (rootIsVerb && root.equals(node)) {
                potentialDeletions = verbAlternates(root, semanticGraph, nerTypes, subTree);
                children = new ArrayList<>(subTree.get(node));
            } else {
                children = semanticGraph.getChildList(node);
            }
            boolean foundFirstModifier = false;

            for (IndexedWord child : children) {
                if ((rootCheck && root.equals(node)) || (!isSplitPoint(node, nerTypes) && !subTree.containsKey(node))) {
                    search.offer(child);
                } else if (isSplitPoint(node, nerTypes) || subTree.containsKey(node)) {
                    boolean modifierCheck = (nodeIsVerb && subTree.containsKey(node)) || isModifier(semanticGraph.getEdge(node, child));
                    if (!foundFirstModifier && modifierCheck) foundFirstModifier = true;
                    if (root.equals(node)) {
                        if (child.index() >= index || foundFirstModifier) search.offer(child);
                    } else {
                        if (child.index() <= newIndex && !foundFirstModifier) search.offer(child);
                    }
                }
            }
            if (!root.equals(node) && (isSplitPoint(node, nerTypes) || subTree.containsKey(node))) {
                subTree = splitNoun(node, subTree, semanticGraph, nerTypes);
            }
        }
        if (potentialDeletions != null) subTree.get(root).removeAll(potentialDeletions);
        return subTree;
    }

    private Map<IndexedWord, String> encodeTree(SemanticGraph semanticGraph, IndexedWord r, String coding, Map<IndexedWord, String> map) {
        if (map.containsKey(r)) return map;
        map.put(r, coding);
        int i = 0;
        for (SemanticGraphEdge edge : semanticGraph.getOutEdgesSorted(r)) {
            i = (i == 26) ? 97 : i;
            char ch = (char) ((int) 'A' + i++);
            String new_coding = coding.split("_")[0] + ch + '_' + edge.getRelation();
            map = encodeTree(semanticGraph, edge.getDependent(), new_coding, map);
        }
        return map;
    }

    private void generateReverseWordEncoding(Map<IndexedWord, String> encodeTree, Map<String, String> entityDict, String[] nerTypes) {
        for(Map.Entry<IndexedWord, String> entry : encodeTree.entrySet()){
            boolean containsEntity = containsEntity(entry.getKey().word(), nerTypes);
            SubSentWords word = new SubSentWords(entry.getKey(), entry.getValue(), containsEntity, entityDict);
            this.reverseEncoding.put(entry.getValue(), word);
        }
    }

    public Map<String, SubSentWords> getReverseWordEncoding() {
        return this.reverseEncoding;
    }

    private void sentenceBreakdown(Map<IndexedWord, String> encodeTree, Map<IndexedWord, TreeSet<IndexedWord>> subTree) {
        for (IndexedWord key : subTree.keySet()) {
            SubSentWords subRoot = this.reverseEncoding.get(encodeTree.get(key));
            List<IndexedWord> values = sort_topToLeaf(subTree.get(key), encodeTree);
            TreeSet<SubSentWords> temp = new TreeSet<>();
            int irregularityCount = 0;
            for (IndexedWord w : values) {
                SubSentWords tempWord = this.reverseEncoding.get(encodeTree.get(w));
                String parentEncode = tempWord.getTrimmedEncoding().substring(0, tempWord.getTrimmedEncoding().length() - 1);
                if (temp.parallelStream().noneMatch(s -> s.getTrimmedEncoding().equals(parentEncode)) && !w.equals(key)) {
                    SubSentWords modifyWord = new SubSentWords(tempWord);
                    modifyWord.setEncoding(subRoot.getTrimmedEncoding() + irregularityCount++ + '_' + tempWord.getEncoding().split("_")[1]);
                    this.reverseEncoding.put(modifyWord.getEncoding(), modifyWord);
                    this.sentenceBreakdown.put(modifyWord, new ArrayList<>(Collections.singletonList(modifyWord)));
                    temp.add(modifyWord);
                } else {
                    temp.add(tempWord);
                }
            }
            this.sentenceBreakdown.put(subRoot, new ArrayList<>(temp));
        }
    }

    public Map<SubSentWords, List<SubSentWords>> getSentenceBreakdown() {
        return this.sentenceBreakdown;
    }

    private void setSentenceAndID(String text, String id) {
        this.sentence = text;
        this.sentID = id;
    }

    public void pushUpSentences(SubSentWords encode, List<SubSentWords> value) {
        this.pushedUpSentences.put(encode, value);
    }

    public Map<SubSentWords, List<SubSentWords>> getPushedUpSentences() {
        return this.pushedUpSentences;
    }

    public Map<String, SubSentWords> getReplaceSurfaceName() {
        return this.replaceSurfaceName;
    }
    public void updateReplacedSurfaceName(String key, SubSentWords value) {
        this.replaceSurfaceName.put(key, value);
    }

    public void resetPushUp() {
        this.replaceSurfaceName.clear();
        this.pushedUpSentences.clear();
    }

    public String getSentID() {
        return this.sentID;
    }
}
