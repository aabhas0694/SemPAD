package cpww;

import java.util.*;
import java.util.stream.Collectors;

public class MetaPattern {
    private String metaPattern;
    private String clippedMetaPattern;
    private int nerCount;
    private boolean valid = true;
    private Set<String> entityContainingWords = new HashSet<>();
    private List<String> entities = new ArrayList<>();
    private Integer frequency = 0;

    MetaPattern(String pat, String[] nerTypes, List<String> stopWords, int frequency) {
        this.setMetaPattern(pat);
        this.setNerCount(this.noOfEntities(nerTypes));
        this.setClippedMetaPattern(stopWords);
        this.setFrequency(frequency);
    }

    private int noOfEntities(String[] nerTypes) {
        int nerNo = 0;
        String[] splitPattern = this.metaPattern.split(" ");

        for (String pat : splitPattern) {
            for (String ner : nerTypes) {
                if (pat.contains(ner)) {
                    this.entityContainingWords.add(pat);
                    this.entities.add(ner);
                    nerNo++;
                    break;
                }
            }
        }
        return nerNo;
    }

    private void setFrequency(Integer frequency) {
        this.frequency = frequency;
    }

    public Integer getFrequency() {
        return this.frequency;
    }

    private void setMetaPattern(String mp) {
        this.metaPattern = mp;
    }

    private void setNerCount(int count) {
        this.nerCount = count;
    }

    /**
    Returns true if there exist conjunctions in a pattern
     */
    private boolean checkConjunction(String pattern) {
        List<String> splitPattern = Arrays.asList(pattern.split(" "));
        return splitPattern.contains("and") || splitPattern.contains("or") || splitPattern.contains("but");
    }

    private void setClippedMetaPattern(List<String> stopWords) {
        if (this.getNerCount() == 0) {
            this.valid = false;
            this.clippedMetaPattern = null;
            return;
        }

        List<String> splitPattern = Arrays.asList(this.metaPattern.split(" ")).stream().
                filter(word -> !word.matches("[`'\\.,:;\\!\\-\\|\"]+|\\-lrb\\-|\\-lsb\\-|\\-rrb\\-|\\-rsb\\-"))
                .collect(Collectors.toList());

        int startIndex = 0, endIndex = splitPattern.size() - 1;
        boolean foundStart = false, foundEnd = false;
        while (startIndex < endIndex && (!foundStart || !foundEnd)) {
            if (!foundStart) {
                if (!stopWords.contains(splitPattern.get(startIndex))) {
                    foundStart = true;
                } else {
                    startIndex++;
                }
            }
            if (!foundEnd) {
                if (!stopWords.contains(splitPattern.get(endIndex))) {
                    foundEnd = true;
                } else {
                    endIndex--;
                }
            }
        }
        if (startIndex >= endIndex || (this.entityContainingWords.size() < this.nerCount &&
                endIndex - startIndex < this.nerCount)) {
            this.valid = false;
            this.clippedMetaPattern = null;
            return;
        }
        this.clippedMetaPattern = String.join(" ", splitPattern.subList(startIndex, endIndex + 1));
        if (this.checkConjunction(this.clippedMetaPattern)) {
            this.valid = false;
        }
    }

    public boolean isValid() {
        return this.valid;
    }

    String getClippedMetaPattern() {
        return clippedMetaPattern;
    }
    
    public Set<String> getEntityContainingWords() {
        return this.entityContainingWords;
    }
    
    public List<String> getEntities() {
        return this.entities;
    }

    int getNerCount() {
        return nerCount;
    }
}
