package cpww;

import cpww.utils.PatternMatchIndices;

import java.util.*;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static cpww.utils.Util.containsEntity;
import static cpww.utils.Util.trimEncoding;

public class PatternInstance {
    private String sentID;
    private String metaPattern;
    private List<SubSentWords> entities = new ArrayList<>();
    private List<Integer> allElementIndices = new ArrayList<>();
    private List<List<SubSentWords>> alternateEntities = new ArrayList<>();
    private StringBuilder sentenceInstance;

    PatternInstance (String sentID, String metaPattern) {
        this.sentID = sentID;
        this.metaPattern = metaPattern;
    }

    PatternInstance (SentenceProcessor sentence, SubSentWords subRoot, String metaPattern, PatternMatchIndices entityPos,
                     String[] nerTypes) {
        this.metaPattern = metaPattern;
        this.sentID = sentence.getSentID();
        this.allElementIndices = entityPos.getElementIndices();
        Map<String, SubSentWords> encodingMap = sentence.getReverseWordEncoding();
        List<SubSentWords> value = sentence.getSentenceBreakdown().get(subRoot);
        List<String> subEnc = value.stream().map(SubSentWords::getEncoding).collect(Collectors.toList());
        String encoding = String.join(" ", subEnc);
        for (Integer entityPo : entityPos.getEntityIndices()) {
            String entity_code = subEnc.get(entityPo);
            if (sentence.getReplaceSurfaceName().containsKey(trimEncoding(entity_code)) &&
                    !trimEncoding(entity_code).equals(subRoot.getTrimmedEncoding())) {
                List<String> output = hierarchical_expansion(sentence, encoding, entity_code);
                this.entities.add(encodingMap.get(output.get(0)));
                this.alternateEntities.add(conjunctionSearch(output.get(0), sentence));
                encoding = output.get(1);
            } else {
                this.entities.add(encodingMap.get(entity_code));
                this.alternateEntities.add(conjunctionSearch(entity_code, sentence));
            }
        }
        sentenceInstance = new StringBuilder();
        sentenceInstance.append(map_original_tokens(sentence, encoding, nerTypes, false)).append("\t");
        sentenceInstance.append(map_original_tokens(sentence, encoding, nerTypes, true));
    }

    private static List<SubSentWords> conjunctionSearch(String encoding, SentenceProcessor sentence) {
        SubSentWords subRoot = sentence.getReverseWordEncoding().get(encoding);
        String encode = encoding.split("_")[0].replaceAll("\\[", "\\\\[");
        List<SubSentWords> entitySent = sentence.getSentenceBreakdown().get(subRoot);
        List<SubSentWords> ans = new ArrayList<>();
        String ccPattern = "^" + encode + "[\\w]_cc";
        if (entitySent.parallelStream().anyMatch(sw -> Pattern.matches(ccPattern, sw.getEncoding())
                && (sw.getLemma().equals("but") || sw.getLemma().equals("between")))) {
            return null;
        }
        if (subRoot.getEncoding().contains("_conj")) {
            SubSentWords ancestor = sentence.getReverseWordEncoding().getOrDefault(encode.substring(0, encode.length() - 1), null);
            if (ancestor != null && ancestor.getLemma().equals(subRoot.getLemma())){
                ans.add(ancestor);
                encode = ancestor.getTrimmedEncoding();
                entitySent = sentence.getSentenceBreakdown().get(ancestor);
            }
        }
        String conjPattern = "^" + encode + "[\\w]_conj";
        for (SubSentWords sw : entitySent) {
            if (!sw.equals(subRoot) && Pattern.matches(conjPattern, sw.getEncoding()) && sw.getLemma().equals(subRoot.getLemma()) &&
            sentence.getSentenceBreakdown().get(sw).size() == 1) {
                ans.add(sw);
            }
        }
        return ans.isEmpty() ? null : ans;
    }

    private static List<String> hierarchical_expansion(SentenceProcessor sentence, String original_subEnc,
                                                       String entityEncode){
        List<SubSentWords> toBeAddedSubSent = sentence.getSentenceBreakdown().get(sentence.getReverseWordEncoding().get(entityEncode));
        String added_subEnc = toBeAddedSubSent.stream().map(SubSentWords::getEncoding).collect(Collectors.joining(" "));
        SubSentWords replace_encode = sentence.getReplaceSurfaceName().get(trimEncoding(entityEncode));
        int index_diff = original_subEnc.length() - entityEncode.length();

        List<String> ans = new ArrayList<>();
        ans.add(replace_encode.getEncoding());
        if (Arrays.asList(original_subEnc.replaceAll("[{}]", "").split(" ")).contains(replace_encode.getEncoding())
                || replace_encode.getEncoding().equals(entityEncode)) {
            // Do nothing
        } else if (original_subEnc.contains(" " + entityEncode + " ")) {
            original_subEnc = original_subEnc.replace(" " + entityEncode + " ", " {{" + added_subEnc + "}} ");
        } else if (original_subEnc.contains("{{" + entityEncode + " ")) {
            original_subEnc = original_subEnc.replace("{{" + entityEncode + " ", "{{{{" + added_subEnc + "}} ");
        } else if (original_subEnc.contains(" " + entityEncode + "}}")) {
            original_subEnc = original_subEnc.replace(" " + entityEncode + "}}", " {{" + added_subEnc + "}}}}");
        } else if (original_subEnc.contains("{{" + entityEncode + "}}")) {
            original_subEnc = original_subEnc.replace("{{" + entityEncode + "}}", "{{{{" + added_subEnc + "}}}}");
        } else if (index_diff > 0 && original_subEnc.substring(index_diff - 1).contains(" " + entityEncode)) {
            original_subEnc = original_subEnc.substring(0, index_diff - 1) + " {{" + added_subEnc + "}}";
        } else if (original_subEnc.contains(entityEncode + " ")) {
            original_subEnc = "{{" + added_subEnc + "}} " + original_subEnc.substring(entityEncode.length() + 1);
        }
        if (!sentence.getReplaceSurfaceName().containsKey(replace_encode.getTrimmedEncoding()) ||
                replace_encode.getEncoding().equals(entityEncode)) {
            ans.add(original_subEnc);
            return ans;
        }
        return hierarchical_expansion(sentence, original_subEnc, replace_encode.getEncoding());
    }

    private String map_original_tokens(SentenceProcessor sentence, String encode, String[] nerTypes, boolean wantLemma){
        Map<String, SubSentWords> encodingMap = sentence.getReverseWordEncoding();
        String[] enc = encode.split(" ");
        List<String> ans = new ArrayList<>();
        for (String encoding : enc){
            StringBuilder temp = new StringBuilder(), res = new StringBuilder(), finalRes = new StringBuilder();
            boolean flag = false;
            for (int i = 0; i < encoding.length(); i++){
                if (res.length() == 0 && encoding.charAt(i) == '{'){
                    temp.append("{");
                }
                else if(res.length() != 0 && encoding.charAt(i) == '}' && !flag){
                    SubSentWords word = encodingMap.get(res.toString());
                    finalRes.append(temp);
                    if (wantLemma && !containsEntity(word.getLemma(), nerTypes)) {
                        finalRes.append(word.getLemma()).append("}");
                    } else {
                        finalRes.append(word.getOriginalWord()).append("}");
                    }
                    flag = true;
                }
                else if (res.length() != 0 && encoding.charAt(i) == '}' && flag) finalRes.append("}");
                else res.append(encoding.charAt(i));
            }
            if (!flag) {
                SubSentWords word = encodingMap.get(res.toString());
                finalRes.append(temp);
                if (wantLemma && !containsEntity(word.getLemma(), nerTypes)) {
                    finalRes.append(word.getLemma());
                } else {
                    finalRes.append(word.getOriginalWord());
                }
            }
            ans.add(finalRes.toString());
        }
        return String.join(" ", ans);
    }

    public List<PatternInstance> generateAlternatePattern() {
        List<PatternInstance> altPatternInstances = new ArrayList<>();
        if (!this.alternateEntities.isEmpty()) {
            PatternInstance altPatternInstance = new PatternInstance(sentID, metaPattern);
            for (int i = 0; i < alternateEntities.size(); i++) {
                if (alternateEntities.get(i) != null) {
                    for (int j = 0; j < alternateEntities.get(i).size(); j++) {
                        Set<Integer> copyIndices = new HashSet<>(allElementIndices);
                        copyIndices.remove(entities.get(i).getIndex());
                        copyIndices.add(alternateEntities.get(i).get(j).getIndex());
                        List<SubSentWords> temp = entities.subList(0, i);
                        temp.add(alternateEntities.get(i).get(j));
                        if (i + 1 != entities.size()) {
                            temp.addAll(entities.subList(i + 1, entities.size()));
                        }
                        altPatternInstance.entities = temp;
                        altPatternInstance.allElementIndices = copyIndices.stream().sorted().collect(Collectors.toList());
                        altPatternInstance.sentenceInstance = new StringBuilder(sentenceInstance.toString().replace(entities.get(i).getOriginalWord(), alternateEntities.get(i).get(j).getOriginalWord()));
                        altPatternInstances.add(altPatternInstance);
                    }
                }
            }
        }
        return altPatternInstances;
    }

    public String toString() {
        if (metaPattern == null) return null;
        StringBuilder output = new StringBuilder(sentID);
        output.append("\t").append(metaPattern).append("\t[")
                .append(entities.stream().map(SubSentWords::getOriginalWord).collect(Collectors.joining(", ")))
                .append("]\t").append(sentenceInstance).append("\n");
        return output.toString();
    }

    public List<Integer> getAllElementIndices() {
        return allElementIndices;
    }

    public List<SubSentWords> getEntities() {
        return entities;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        PatternInstance that = (PatternInstance) o;
        if (this.allElementIndices.size() != that.allElementIndices.size()) return false;
        IntStream iStream = IntStream.range(0, allElementIndices.size()).parallel();
        boolean notSame = iStream.anyMatch(i -> !allElementIndices.get(i).equals(that.allElementIndices.get(i)));
        iStream.close();
        return sentID.equals(that.sentID) &&
                metaPattern.equals(that.metaPattern) &&
                !notSame;
    }
}
