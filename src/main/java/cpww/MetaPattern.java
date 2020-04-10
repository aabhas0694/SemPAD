package cpww;

import java.util.*;
import java.util.stream.Collectors;

public class MetaPattern {
    private String metaPattern;
    private String clippedMetaPattern;
    private boolean valid = true;
    private Set<String> entityContainingWords = new HashSet<>();
    private List<Integer> entityIndex = new ArrayList<>();
    private Integer frequency = 0;

    MetaPattern(String pat, String[] nerTypes, List<String> stopWords, int frequency) {
        this.processPattern(pat, nerTypes);
        this.setClippedMetaPattern(stopWords);
        this.setFrequency(frequency);
    }

    private void processPattern(String pattern, String[] nerTypes) {
       this.metaPattern = Arrays.stream(pattern.split(" "))
               .filter(word -> !word.matches("[`$\\d'\\.,\\*:;\\!\\-\\|\"\\\\]+(lrb|lsb|rrb|rsb)*[`$\\d'\\.,\\*:;\\!\\-\\|\"\\\\]*"))
               .collect(Collectors.joining(" "));

        String[] splitPattern = this.metaPattern.split(" ");
        for (int i= 0; i < splitPattern.length; i++) {
            String pat = splitPattern[i];
            for (String ner : nerTypes) {
                if (pat.contains(ner)) {
                    this.entityContainingWords.add(pat);
                    this.entityIndex.add(i);
                    break;
                }
            }
        }
    }

    private void setFrequency(Integer frequency) {
        this.frequency = frequency;
    }

    public Integer getFrequency() {
        return this.frequency;
    }

    /**
    Returns true if there exist conjunctions in a pattern
     */
    private boolean checkConjunction(String pattern) {
        List<String> splitPattern = Arrays.asList(pattern.split(" "));
        return ((splitPattern.contains("and") || splitPattern.contains("or"))  && !splitPattern.contains("between")) || splitPattern.contains("but");
    }

    /**
     * Checks if all entities are not placed one after the other at consecutive positions.
     * @return If so, returns true, else false.
     */
    private boolean isEntityContinuous() {
        if (getNerCount() == 0) return true;
        else if (getNerCount() == 1) return false;
        String[] splitPattern = this.metaPattern.split(" ");
        for (int i = 0; i < entityIndex.size() - 1; i++) {
            int i1 = entityIndex.get(i), i2 = entityIndex.get(i + 1);
            if (i1 + 1 == i2 && splitPattern[i1].equals(splitPattern[i2])) {
                return true;
            }
        }
        return false;
    }

    private void setClippedMetaPattern(List<String> stopWords) {
        if (isEntityContinuous()) {
            this.valid = false;
            this.clippedMetaPattern = null;
            return;
        }

        String[] splitPattern = this.metaPattern.split(" ");
        int startIndex = 0, endIndex = splitPattern.length - 1;
        boolean foundStart = false, foundEnd = false;
        while (startIndex < endIndex && (!foundStart || !foundEnd)) {
            if (!foundStart) {
                if (!stopWords.contains(splitPattern[startIndex])) {
                    foundStart = true;
                } else {
                    startIndex++;
                }
            }
            if (!foundEnd) {
                if (!stopWords.contains(splitPattern[endIndex])) {
                    foundEnd = true;
                } else {
                    endIndex--;
                }
            }
        }
        if (startIndex >= endIndex || (this.entityContainingWords.size() < getNerCount() &&
                endIndex - startIndex < getNerCount()) || getNerCount() > 1 && endIndex - startIndex == getNerCount() - 1) {
            this.valid = false;
            this.clippedMetaPattern = null;
            return;
        }
        this.clippedMetaPattern = String.join(" ", Arrays.copyOfRange(splitPattern, startIndex, endIndex + 1));
        if (this.checkConjunction(this.clippedMetaPattern)) {
            this.valid = false;
        }
    }

    public boolean isValid() {
        return this.valid;
    }

    public String getClippedMetaPattern() {
        return clippedMetaPattern;
    }
    
    public int getNerCount() {
        return entityIndex.size();
    }
}
