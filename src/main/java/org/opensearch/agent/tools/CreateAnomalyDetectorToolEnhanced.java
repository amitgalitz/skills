/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.agent.tools;

import static org.opensearch.agent.tools.utils.CommonConstants.COMMON_MODEL_ID_FIELD;
import static org.opensearch.ml.common.CommonValue.TENANT_ID_FIELD;
import static org.opensearch.ml.common.utils.StringUtils.gson;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.StringJoiner;
import java.util.concurrent.CompletableFuture;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import org.apache.commons.text.StringSubstitutor;
import org.opensearch.action.ActionRequest;
import org.opensearch.action.admin.indices.mapping.get.GetMappingsRequest;
import org.opensearch.action.search.SearchRequest;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.ad.client.AnomalyDetectionNodeClient;
import org.opensearch.ad.model.AnomalyDetector;
import org.opensearch.ad.transport.IndexAnomalyDetectorRequest;
import org.opensearch.agent.tools.utils.ToolHelper;
import org.opensearch.cluster.metadata.MappingMetadata;
import org.opensearch.common.UUIDs;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.common.Strings;
import org.opensearch.core.common.io.stream.NamedWriteableRegistry;
import org.opensearch.index.query.QueryBuilders;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.dataset.remote.RemoteInferenceInputDataSet;
import org.opensearch.ml.common.input.MLInput;
import org.opensearch.ml.common.output.model.ModelTensor;
import org.opensearch.ml.common.output.model.ModelTensorOutput;
import org.opensearch.ml.common.output.model.ModelTensors;
import org.opensearch.ml.common.spi.tools.ToolAnnotation;
import org.opensearch.ml.common.spi.tools.WithModelTool;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskAction;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskRequest;
import org.opensearch.ml.common.utils.ToolUtils;
import org.opensearch.rest.RestRequest;
import org.opensearch.search.aggregations.AggregationBuilder;
import org.opensearch.search.aggregations.AggregationBuilders;
import org.opensearch.search.builder.SearchSourceBuilder;
import org.opensearch.timeseries.AnalysisType;
import org.opensearch.timeseries.model.Feature;
import org.opensearch.timeseries.model.IntervalTimeConfiguration;
import org.opensearch.timeseries.model.TimeConfiguration;
import org.opensearch.timeseries.transport.JobRequest;
import org.opensearch.timeseries.transport.SuggestConfigParamRequest;
import org.opensearch.timeseries.transport.SuggestConfigParamResponse;
import org.opensearch.timeseries.transport.ValidateConfigRequest;
import org.opensearch.timeseries.transport.ValidateConfigResponse;
import org.opensearch.transport.client.Client;

import com.google.common.collect.ImmutableMap;

import lombok.Getter;
import lombok.Setter;
import lombok.extern.log4j.Log4j2;

/**
 * A tool used to help creating anomaly detector, the only one input parameter is the index name, this tool will get the mappings of the index
 * in flight and let LLM give the suggested category field, aggregation field and correspond aggregation method which are required for the create
 * anomaly detector API, the output of this tool is like:
 *{
 *     "index": "opensearch_dashboards_sample_data_ecommerce",
 *     "categoryField": "geoip.country_iso_code",
 *     "aggregationField": "total_quantity,total_unique_products,taxful_total_price",
 *     "aggregationMethod": "sum,count,sum",
 *     "dateFields": "customer_birth_date,order_date,products.created_on"
 * }
 */
@Log4j2
@Setter
@Getter
@ToolAnnotation(CreateAnomalyDetectorToolEnhanced.TYPE)
public class CreateAnomalyDetectorToolEnhanced implements WithModelTool {
    // the type of this tool
    public static final String TYPE = "CreateAnomalyDetectorToolEnhanced";

    // the default description of this tool
    private static final String DEFAULT_DESCRIPTION =
        "Enhanced tool for creating anomaly detector configurations. Takes an index name, extracts the index mappings, and uses LLM to generate complete detector JSON configurations ready for the create detector API.";
    // the regex used to extract the key information from the response of LLM
    private static final String EXTRACT_INFORMATION_REGEX =
        "(?s).*\\{category_field=([^|]*)\\|aggregation_field=([^|]*)\\|aggregation_method=([^|]*)\\|interval=([^}]*)}.*";
    // valid field types which support aggregation
    private static final Set<String> VALID_FIELD_TYPES = Set
        .of(
            "keyword",
            "constant_keyword",
            "wildcard",
            "long",
            "integer",
            "short",
            "byte",
            "double",
            "float",
            "half_float",
            "scaled_float",
            "unsigned_long",
            "ip"
        );
    // the index name key in the output
    private static final String OUTPUT_KEY_INDEX = "index";
    // the category field key in the output
    private static final String OUTPUT_KEY_CATEGORY_FIELD = "categoryField";
    // the aggregation field key in the output
    private static final String OUTPUT_KEY_AGGREGATION_FIELD = "aggregationField";
    // the aggregation method name key in the output
    private static final String OUTPUT_KEY_AGGREGATION_METHOD = "aggregationMethod";
    // the date fields key in the output
    private static final String OUTPUT_KEY_DATE_FIELDS = "dateFields";
    // the default prompt dictionary, includes claude and openai
    private static final Map<String, String> DEFAULT_PROMPT_DICT = loadDefaultPromptFromFile();

    // Retry limits for different phases
    private static final int MAX_DETECTOR_VALIDATION_RETRIES = 3;
    private static final int MAX_MODEL_VALIDATION_RETRIES = 3;
    private static final int MAX_FORMAT_FIX_RETRIES = 1;

    // Detector configuration defaults
    private static final int DEFAULT_INTERVAL_MINUTES = 10;
    private static final int DEFAULT_WINDOW_DELAY_MINUTES = 1;
    private static final int DEFAULT_SHINGLE_SIZE = 8;
    private static final int DEFAULT_SCHEMA_VERSION = 1;
    private static final String DEFAULT_DETECTOR_DESCRIPTION = "Detector generated by OpenSearch Assistant";

    // Interval thresholds for sparse data handling
    private static final int UNREASONABLE_INTERVAL_THRESHOLD_MINUTES = 240; // 4 hours
    private static final int SPARSE_DATA_WARNING_THRESHOLD_MINUTES = 120; // 2 hours

    // the name of this tool
    @Setter
    @Getter
    private String name = TYPE;
    // the description of this tool
    @Getter
    @Setter
    private String description = DEFAULT_DESCRIPTION;

    // the version of this tool
    @Getter
    private String version;

    // the OpenSearch transport client
    private Client client;
    // the anomaly detection node client for validation
    private AnomalyDetectionNodeClient adClient;
    // the mode id of LLM
    @Getter
    private String modelId;
    // LLM model type, CLAUDE or OPENAI
    @Getter
    private ModelType modelType;
    // the default prompt for creating anomaly detector
    private String contextPrompt;
    private Map<String, Object> attributes;

    enum ModelType {
        CLAUDE,
        OPENAI;

        public static ModelType from(String value) {
            return valueOf(value.toUpperCase(Locale.ROOT));
        }

    }

    /**
     *
     * @param client the OpenSearch transport client
     * @param modelId the model ID of LLM
     * @param modelType the model type (CLAUDE or OPENAI)
     * @param contextPrompt custom prompt (if empty, loads from file)
     * @param promptFile prompt file name (without .json extension, empty for default)
     * @param namedWriteableRegistry the named writeable registry
     */
    public CreateAnomalyDetectorToolEnhanced(
        Client client,
        String modelId,
        String modelType,
        String contextPrompt,
        String promptFile,
        NamedWriteableRegistry namedWriteableRegistry
    ) {
        this.client = client;
        this.adClient = new AnomalyDetectionNodeClient(client, namedWriteableRegistry);
        this.modelId = modelId;
        if (!ModelType.OPENAI.toString().equalsIgnoreCase(modelType) && !ModelType.CLAUDE.toString().equalsIgnoreCase(modelType)) {
            throw new IllegalArgumentException("Unsupported model_type: " + modelType);
        }
        this.modelType = ModelType.from(modelType);

        if (contextPrompt.isEmpty()) {
            if (promptFile.isEmpty()) {
                // Use cached default prompts
                this.contextPrompt = DEFAULT_PROMPT_DICT.getOrDefault(this.modelType.toString(), "");
            } else {
                // Load custom prompt file dynamically
                Map<String, String> customPrompts = loadCustomPromptFromFile(promptFile);
                this.contextPrompt = customPrompts.getOrDefault(this.modelType.toString(), "");
            }
        } else {
            this.contextPrompt = contextPrompt;
        }
    }

    /**
     * The main running method of this tool
     * @param parameters the input parameters
     * @param listener the action listener
     */
    @Override
    public <T> void run(Map<String, String> parameters, ActionListener<T> listener) {
        if (parameters.containsKey("input")) {
            String inputStr = parameters.get("input");
            if (inputStr != null && inputStr.trim().startsWith("[")) {
                parameters.put("input", "{\"indices\": " + inputStr + "}");
            }
        }
        parameters = ToolUtils.extractInputParameters(parameters, attributes);
        final String tenantId = parameters.get(TENANT_ID_FIELD);
        try {
            List<String> indices = extractIndicesList(parameters);
            int maxRetries = Integer.parseInt(parameters.getOrDefault("maxRetries", "1"));

            processMultipleIndices(indices, tenantId, maxRetries, listener);
        } catch (Exception e) {
            listener.onFailure(e);
        }
    }

    private List<String> extractIndicesList(Map<String, String> parameters) {
        String inputStr = parameters.get("input");
        if (inputStr == null || inputStr.trim().isEmpty()) {
            throw new IllegalArgumentException("Input parameter is required");
        }

        try {
            Map<String, Object> input = gson.fromJson(inputStr, Map.class);
            List<String> indices = (List<String>) input.get("indices");

            if (indices == null || indices.isEmpty()) {
                throw new IllegalArgumentException("No indices provided");
            }

            for (String index : indices) {
                if (index.startsWith(".")) {
                    throw new IllegalArgumentException("System indices not supported: " + index);
                }
            }

            return indices;
        } catch (Exception e) {
            throw new IllegalArgumentException("Failed to parse indices: " + e.getMessage());
        }
    }

    private <T> void processMultipleIndices(List<String> indices, String tenantId, int maxRetries, ActionListener<T> listener) {
        log.info("Processing {} indices sequentially for detector creation", indices.size());
        Map<String, String> results = new HashMap<>();
        processNextIndex(indices, 0, tenantId, maxRetries, results, listener);
    }

    private <T> void processNextIndex(
        List<String> indices,
        int currentIndex,
        String tenantId,
        int maxRetries,
        Map<String, String> results,
        ActionListener<T> listener
    ) {
        if (currentIndex >= indices.size()) {
            log.info("Completed processing all {} indices", indices.size());
            listener.onResponse((T) gson.toJson(results));
            return;
        }

        String indexName = indices.get(currentIndex);
        log.info("Processing index {}/{}: {}", currentIndex + 1, indices.size(), indexName);

        processSingleIndex(indexName, tenantId, maxRetries, new ActionListener<String>() {
            @Override
            public void onResponse(String result) {
                results.put(indexName, result);
                processNextIndex(indices, currentIndex + 1, tenantId, maxRetries, results, listener);
            }

            @Override
            public void onFailure(Exception e) {
                results.put(indexName, DetectorResult.failedValidation(indexName, e.getMessage()).toJson());
                processNextIndex(indices, currentIndex + 1, tenantId, maxRetries, results, listener);
            }
        });
    }

    private void processSingleIndex(String indexName, String tenantId, int maxRetries, ActionListener<String> listener) {
        getMappingsAndFilterFields(
            indexName,
            ActionListener
                .wrap(
                    mappingContext -> generateAndParseConfig(
                        mappingContext,
                        tenantId,
                        maxRetries,
                        listener,
                        (suggestions, listenerCallback) -> validateDetectorPhase(suggestions, tenantId, maxRetries, 0, listenerCallback)
                    ),
                    listener::onFailure
                )
        );
    }

    /**
     * Extract indices from input parameter
     * Supports two formats:
     * 1. Array: ["index1", "index2"]
     * 2. Object: {"indices": ["index1", "index2"]}
     * @param parameters the original parameters
     * @return parameters with "index" set to first index from the list
     */
    private Map<String, String> enrichParameters(Map<String, String> parameters) {
        Map<String, String> result = new HashMap<>(parameters);
        try {
            String inputStr = parameters.get("input");
            // Try to parse as array first (frontend format)
            if (inputStr.trim().startsWith("[")) {
                List<String> indices = gson.fromJson(inputStr, List.class);
                if (indices != null && !indices.isEmpty()) {
                    result.put("index", indices.get(0));
                }
            } else {
                Map<String, Object> input = gson.fromJson(inputStr, Map.class);
                List<String> indices = (List<String>) input.get("indices");
                if (indices != null && !indices.isEmpty()) {
                    result.put("index", indices.get(0));
                }
            }
        } catch (Exception e) {
            log.error("Failed to parse input parameter: " + e.getMessage(), e);
        }
        return result;
    }

    private String extractResponseFromDataAsMap(Map<String, Object> dataAsMap) {
        if (dataAsMap == null) {
            log.error("dataAsMap is null");
            return null;
        }

        if (dataAsMap.containsKey("response")) {
            String response = (String) dataAsMap.get("response");
            log.info("Found direct response: {}", response);
            return response;
        } else if (dataAsMap.containsKey("output")) {
            // Parse Bedrock format: output.message.content[0].text
            try {
                Map<String, Object> output = (Map<String, Object>) dataAsMap.get("output");
                Map<String, Object> message = (Map<String, Object>) output.get("message");
                List<Map<String, Object>> content = (List<Map<String, Object>>) message.get("content");
                String response = (String) content.get(0).get("text");
                log.info("Extracted Bedrock response: {}", response);
                return response;
            } catch (Exception e) {
                log.error("Failed to parse Bedrock response format", e);
                return null;
            }
        } else {
            log.error("Unknown response format. Available keys: {}", dataAsMap.keySet());
            return null;
        }
    }

    /**
     *
     * @param fieldsToType the flattened field-> field type mapping
     * @return a list containing all the date type fields
     */
    private Set<String> findDateTypeFields(final Map<String, String> fieldsToType) {
        Set<String> result = new HashSet<>();
        for (Map.Entry<String, String> entry : fieldsToType.entrySet()) {
            String value = entry.getValue();
            if (value.equals("date") || value.equals("date_nanos")) {
                result.add(entry.getKey());
            }
        }
        return result;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, String> loadDefaultPromptFromFile() {
        try (
            InputStream inputStream = CreateAnomalyDetectorToolEnhanced.class
                .getResourceAsStream("CreateAnomalyDetectorEnhancedPromptV5.json")
        ) {
            if (inputStream != null) {
                return gson.fromJson(new String(inputStream.readAllBytes(), StandardCharsets.UTF_8), Map.class);
            }
        } catch (IOException e) {
            log.error("Failed to load prompt from the file CreateAnomalyDetectorDefaultPromptEnhanced.json, error: ", e);
        }
        return new HashMap<>();
    }

    @SuppressWarnings("unchecked")
    private static Map<String, String> loadCustomPromptFromFile(String promptFile) {
        String fileName = promptFile + ".json";
        try (InputStream inputStream = CreateAnomalyDetectorToolEnhanced.class.getResourceAsStream(fileName)) {
            if (inputStream != null) {
                return gson.fromJson(new String(inputStream.readAllBytes(), StandardCharsets.UTF_8), Map.class);
            }
        } catch (IOException e) {
            log.error("Failed to load prompt from file {}, error: ", fileName, e);
        }
        return new HashMap<>();
    }

    /**
     *
     * @param fieldsToType the flattened field-> field type mapping
     * @param indexName the index name
     * @param dateFields the comma-separated date fields
     * @return the prompt about creating anomaly detector
     */
    private String constructPrompt(final Map<String, String> fieldsToType, final String indexName, final String dateFields) {
        StringJoiner tableInfoJoiner = new StringJoiner("\n");
        for (Map.Entry<String, String> entry : fieldsToType.entrySet()) {
            tableInfoJoiner.add("- " + entry.getKey() + ": " + entry.getValue());
        }

        Map<String, String> indexInfo = ImmutableMap
            .of("indexName", indexName, "indexMapping", tableInfoJoiner.toString(), "dateFields", dateFields);
        StringSubstitutor substitutor = new StringSubstitutor(indexInfo, "${indexInfo.", "}");
        return substitutor.replace(contextPrompt);
    }

    /**
     *
     * @param parameters the input parameters
     * @return false if the input parameters is null or empty
     */
    @Override
    public boolean validate(Map<String, String> parameters) {
        return parameters != null && parameters.size() != 0;
    }

    /**
     *
     * @return the type of this tool
     */
    @Override
    public String getType() {
        return TYPE;
    }

    private AnomalyDetector buildAnomalyDetectorFromSuggestions(Map<String, String> suggestions) {
        String indexName = suggestions.get(OUTPUT_KEY_INDEX);
        String categoryField = suggestions.get(OUTPUT_KEY_CATEGORY_FIELD);
        String aggregationFields = suggestions.get(OUTPUT_KEY_AGGREGATION_FIELD);
        String aggregationMethods = suggestions.get(OUTPUT_KEY_AGGREGATION_METHOD);
        String dateFields = suggestions.get(OUTPUT_KEY_DATE_FIELDS);
        String intervalStr = suggestions.getOrDefault("interval", String.valueOf(DEFAULT_INTERVAL_MINUTES));

        // Parse interval (default to 10 minutes)
        int intervalMinutes = DEFAULT_INTERVAL_MINUTES;
        try {
            intervalMinutes = Integer.parseInt(intervalStr);
        } catch (NumberFormatException e) {
            log.warn("Invalid interval '{}', using default {} minutes", intervalStr, DEFAULT_INTERVAL_MINUTES);
        }

        // Parse comma-separated fields and methods
        String[] fields = aggregationFields.split(",");
        String[] methods = aggregationMethods.split(",");

        if (fields.length != methods.length) {
            throw new IllegalArgumentException("Number of aggregation fields and methods must match");
        }

        // Build features list
        List<Feature> features = new ArrayList<>();
        for (int i = 0; i < fields.length; i++) {
            String field = fields[i].trim();
            String method = methods[i].trim();

            if (field.isEmpty() || method.isEmpty()) {
                continue;
            }

            // Remove "feature_" prefix if LLM already added it to avoid double prefix
            String cleanField = field.startsWith("feature_") ? field.substring(8) : field;

            AggregationBuilder aggregation = createAggregationBuilder(method, cleanField);
            Feature feature = new Feature(UUIDs.randomBase64UUID(), "feature_" + cleanField, true, aggregation);
            features.add(feature);
        }

        // Build category fields list
        List<String> categoryFields = null;
        if (categoryField != null
            && !categoryField.trim().isEmpty()
            && !categoryField.trim().equalsIgnoreCase("null")
            && !categoryField.trim().equalsIgnoreCase("none")) {
            categoryFields = List.of(categoryField.trim());
        }

        // Use first date field as time field
        String timeField = dateFields.split(",")[0].trim();

        return new AnomalyDetector(
            null, // detectorId - null for new detector
            null, // version
            indexName + "-detector-" + UUIDs.randomBase64UUID().substring(0, 8), // name
            DEFAULT_DETECTOR_DESCRIPTION, // description
            timeField, // timeField
            List.of(indexName), // indices
            features, // features
            QueryBuilders.matchAllQuery(), // filterQuery
            new IntervalTimeConfiguration(intervalMinutes, ChronoUnit.MINUTES), // detectionInterval - from suggestions
            new IntervalTimeConfiguration(DEFAULT_WINDOW_DELAY_MINUTES, ChronoUnit.MINUTES), // windowDelay
            DEFAULT_SHINGLE_SIZE, // shingleSize
            null, // uiMetadata
            DEFAULT_SCHEMA_VERSION, // schemaVersion
            Instant.now(), // lastUpdateTime
            categoryFields, // categoryFields
            null, // user
            null, // resultIndex
            null, // imputationOption
            null, // recencyEmphasis
            null, // seasonIntervals
            null, // historyIntervals
            null, // rules
            null, // customResultIndexMinSize
            null, // customResultIndexMinAge
            null, // customResultIndexTTL
            null, // flattenResultIndexMapping
            null, // lastBreakingUIChangeTime
            new IntervalTimeConfiguration(intervalMinutes, ChronoUnit.MINUTES), // frequency - same as detection interval
            true  // autoCreated - true since this is created by the assistant
        );
    }

    private AggregationBuilder createAggregationBuilder(String method, String field) {
        switch (method.toLowerCase(Locale.ROOT)) {
            case "avg":
                return AggregationBuilders.avg(field).field(field);
            case "sum":
                return AggregationBuilders.sum(field).field(field);
            case "min":
                return AggregationBuilders.min(field).field(field);
            case "max":
                return AggregationBuilders.max(field).field(field);
            case "count":
                return AggregationBuilders.count(field).field(field);
            default:
                throw new IllegalArgumentException("Unsupported aggregation method: " + method);
        }
    }

    private <T> void retryWithFormatFix(
        String parseError,
        String indexName,
        String dateFields,
        String tenantId,
        int maxRetries,
        int currentRetry,
        String validationType,
        ActionListener<T> listener,
        java.util.function.BiConsumer<Map<String, String>, ActionListener<T>> nextPhaseCallback
    ) {
        String fixPrompt = "The previous response had incorrect format. "
            + parseError
            + "\n\nPlease provide suggestions for index '"
            + indexName
            + "' in the exact format: "
            + "{category_field=field|aggregation_field=field1,field2|aggregation_method=method1,method2|interval=10}"
            + "\n\nInterval should be in minutes (default: 10). Only return the configuration in curly braces.";

        RemoteInferenceInputDataSet inputDataSet = RemoteInferenceInputDataSet
            .builder()
            .parameters(Collections.singletonMap("prompt", fixPrompt))
            .build();
        ActionRequest request = new MLPredictionTaskRequest(
            modelId,
            MLInput.builder().algorithm(FunctionName.REMOTE).inputDataset(inputDataSet).build(),
            null,
            tenantId
        );

        client.execute(MLPredictionTaskAction.INSTANCE, request, ActionListener.wrap(mlTaskResponse -> {
            ModelTensorOutput modelTensorOutput = (ModelTensorOutput) mlTaskResponse.getOutput();
            ModelTensors modelTensors = modelTensorOutput.getMlModelOutputs().get(0);
            ModelTensor modelTensor = modelTensors.getMlModelTensors().get(0);
            Map<String, Object> dataAsMap = (Map<String, Object>) modelTensor.getDataAsMap();
            String fixedResponse = extractResponseFromDataAsMap(dataAsMap);

            if (Strings.isNullOrEmpty(fixedResponse)) {
                listener.onFailure(new IllegalStateException("Remote endpoint fails to inference during format fix."));
                return;
            }
            parseAndRetryWithLLM(
                fixedResponse,
                indexName,
                dateFields,
                tenantId,
                maxRetries,
                currentRetry,
                validationType,
                listener,
                nextPhaseCallback
            );
        }, e -> {
            log.error("Failed to get LLM format fix: " + e);
            listener.onFailure(e);
        }));
    }

    private <T> void callValidationAPI(AnomalyDetector detector, String validationType, ActionListener<ValidateConfigResponse> listener) {
        // Log detailed detector configuration before validation
        log.info("Calling validation API with type: {}", validationType);
        log.info("Detector configuration for validation:");
        log.info("  - Name: {}", detector.getName());
        log.info("  - Time field: {}", detector.getTimeField());
        log.info("  - Indices: {}", detector.getIndices());
        log.info("  - Detection interval: {} {}", detector.getIntervalInMinutes(), "minutes");
        log.info("  - Category fields: {}", detector.getCategoryFields() == null ? "NONE" : detector.getCategoryFields());
        log
            .info(
                "  - Detector type: {}",
                detector.getCategoryFields() == null || detector.getCategoryFields().isEmpty() ? "SINGLE_ENTITY" : "MULTI_ENTITY"
            );
        log.info("  - Number of features: {}", detector.getFeatureAttributes().size());

        // Log each feature in detail
        for (int i = 0; i < detector.getFeatureAttributes().size(); i++) {
            Feature feature = detector.getFeatureAttributes().get(i);
            log
                .info(
                    "  - Feature {}: name='{}', enabled={}, aggregation={}",
                    i + 1,
                    feature.getName(),
                    feature.getEnabled(),
                    feature.getAggregation().toString()
                );
        }

        try {
            ValidateConfigRequest validateRequest = new ValidateConfigRequest(AnalysisType.AD, detector, validationType);
            adClient.validateAnomalyDetector(validateRequest, listener);
        } catch (Throwable e) {
            log.error("Validation API call failed: {}", e.getMessage(), e);
            listener.onFailure(new RuntimeException("Validation API failed: " + e.getMessage(), e));
        }
    }

    private <T> void parseAndRetryWithLLM(
        String llmResponse,
        String indexName,
        String dateFields,
        String tenantId,
        int maxRetries,
        int currentRetry,
        String validationType,
        ActionListener<T> listener,
        java.util.function.BiConsumer<Map<String, String>, ActionListener<T>> nextPhaseCallback
    ) {
        Pattern pattern = Pattern.compile(EXTRACT_INFORMATION_REGEX);
        Matcher matcher = pattern.matcher(llmResponse);

        log.info("Attempting to parse LLM response: {}", llmResponse);

        if (!matcher.matches()) {
            log.error("Regex parsing failed for response: {}", llmResponse);
            if (currentRetry < maxRetries) {
                String parseError =
                    "Cannot parse response format. Expected: {category_field=field|aggregation_field=field1,field2|aggregation_method=method1,method2|interval=minutes}";
                retryWithFormatFix(
                    parseError,
                    indexName,
                    dateFields,
                    tenantId,
                    maxRetries,
                    currentRetry + 1,
                    validationType,
                    listener,
                    nextPhaseCallback
                );
            } else {
                listener.onFailure(new IllegalStateException("Cannot parse LLM response after " + maxRetries + " retries"));
            }
            return;
        }

        // Parse successful - build suggestions
        String categoryField = matcher.group(1).replaceAll("\"", "").strip();
        String aggregationField = matcher.group(2).replaceAll("\"", "").strip();
        String aggregationMethod = matcher.group(3).replaceAll("\"", "").strip();
        String interval = matcher.group(4).replaceAll("\"", "").strip();

        log.info("Available date fields: {}", dateFields);

        // Select optimal date field from all available date fields
        String[] dateFieldArray = dateFields.split(",");
        selectOptimalDateField(indexName, dateFieldArray, new ActionListener<String>() {
            @Override
            public void onResponse(String optimalDateField) {
                log.info("Using optimal date field: {}", optimalDateField);

                Map<String, String> suggestions = Map
                    .of(
                        OUTPUT_KEY_INDEX,
                        indexName,
                        OUTPUT_KEY_CATEGORY_FIELD,
                        categoryField,
                        OUTPUT_KEY_AGGREGATION_FIELD,
                        aggregationField,
                        OUTPUT_KEY_AGGREGATION_METHOD,
                        aggregationMethod,
                        OUTPUT_KEY_DATE_FIELDS,
                        optimalDateField,
                        "interval",
                        interval
                    );

                nextPhaseCallback.accept(suggestions, listener);
            }

            @Override
            public void onFailure(Exception e) {
                log.error("Failed to select optimal date field, using first available: {}", e.getMessage());
                String fallbackDateField = dateFieldArray.length > 0 ? dateFieldArray[0].trim() : dateFields.split(",")[0];

                Map<String, String> suggestions = Map
                    .of(
                        OUTPUT_KEY_INDEX,
                        indexName,
                        OUTPUT_KEY_CATEGORY_FIELD,
                        categoryField,
                        OUTPUT_KEY_AGGREGATION_FIELD,
                        aggregationField,
                        OUTPUT_KEY_AGGREGATION_METHOD,
                        aggregationMethod,
                        OUTPUT_KEY_DATE_FIELDS,
                        fallbackDateField,
                        "interval",
                        interval
                    );

                nextPhaseCallback.accept(suggestions, listener);
            }
        });
    }

    private <T> void validateDetectorPhase(
        Map<String, String> suggestions,
        String tenantId,
        int maxRetries,
        int currentRetry,
        ActionListener<T> listener
    ) {
        try {
            log.info("Phase 1: Detector validation - Building AnomalyDetector from suggestions: {}", suggestions);
            AnomalyDetector detector = buildAnomalyDetectorFromSuggestions(suggestions);
            log.info("Phase 1: Successfully built AnomalyDetector: {}", detector.getName());

            callValidationAPI(detector, "detector", new ActionListener<ValidateConfigResponse>() {
                @Override
                public void onResponse(ValidateConfigResponse response) {
                    log.info("Phase 1: Detector validation API response: {}", response);
                    log.info("Phase 1: Detector validation API succeeded");
                    log.info("Phase 1: Validation issue: {}", response.getIssue() == null ? "None" : response.getIssue().getMessage());

                    if (response.getIssue() != null) {
                        // All detector validation issues are blocking - retry with LLM fix
                        log.info("Phase 1: Detector validation failed, retrying with LLM fix");
                        String errorMessage = response.getIssue().getMessage();
                        if (currentRetry < MAX_DETECTOR_VALIDATION_RETRIES) {
                            retryDetectorValidation(suggestions, errorMessage, tenantId, maxRetries, currentRetry + 1, listener);
                        } else {
                            log.error("Phase 1: Max detector validation retries reached: {}", errorMessage);
                            listener
                                .onFailure(
                                    new RuntimeException(
                                        "Detector validation failed after " + MAX_DETECTOR_VALIDATION_RETRIES + " retries: " + errorMessage
                                    )
                                );
                        }
                        return;
                    }

                    // Detector validation passed - proceed to suggest phase
                    log.info("Phase 1: Detector validation passed - proceeding to suggest phase");
                    suggestHyperParametersPhase(detector, tenantId, maxRetries, listener);
                }

                @Override
                public void onFailure(Exception e) {
                    log.error("Phase 1: Detector validation API failed: {}", e.getMessage(), e);
                    if (currentRetry < MAX_DETECTOR_VALIDATION_RETRIES) {
                        retryDetectorValidation(suggestions, e.getMessage(), tenantId, maxRetries, currentRetry + 1, listener);
                    } else {
                        listener
                            .onFailure(
                                new RuntimeException(
                                    "Detector validation failed after " + MAX_DETECTOR_VALIDATION_RETRIES + " retries: " + e.getMessage()
                                )
                            );
                    }
                }
            });

        } catch (Exception e) {
            log.error("Phase 1: Error building detector: {}", e.getMessage(), e);
            if (currentRetry < MAX_FORMAT_FIX_RETRIES) {
                retryDetectorValidation(suggestions, e.getMessage(), tenantId, maxRetries, currentRetry + 1, listener);
            } else {
                listener.onFailure(e);
            }
        }
    }

    private <T> void retryDetectorValidation(
        Map<String, String> originalSuggestions,
        String validationError,
        String tenantId,
        int maxRetries,
        int currentRetry,
        ActionListener<T> listener
    ) {
        String fixPrompt = createFixPrompt(originalSuggestions, validationError);

        RemoteInferenceInputDataSet inputDataSet = RemoteInferenceInputDataSet
            .builder()
            .parameters(Collections.singletonMap("prompt", fixPrompt))
            .build();
        ActionRequest request = new MLPredictionTaskRequest(
            modelId,
            MLInput.builder().algorithm(FunctionName.REMOTE).inputDataset(inputDataSet).build(),
            null,
            tenantId
        );

        client.execute(MLPredictionTaskAction.INSTANCE, request, ActionListener.wrap(mlTaskResponse -> {
            ModelTensorOutput modelTensorOutput = (ModelTensorOutput) mlTaskResponse.getOutput();
            ModelTensors modelTensors = modelTensorOutput.getMlModelOutputs().get(0);
            ModelTensor modelTensor = modelTensors.getMlModelTensors().get(0);
            Map<String, Object> dataAsMap = (Map<String, Object>) modelTensor.getDataAsMap();
            String fixedResponse = extractResponseFromDataAsMap(dataAsMap);

            if (Strings.isNullOrEmpty(fixedResponse)) {
                listener.onFailure(new IllegalStateException("Remote endpoint fails to inference during detector validation fix."));
                return;
            }

            // Parse fixed response and retry detector validation
            parseAndRetryWithLLM(
                fixedResponse,
                originalSuggestions.get(OUTPUT_KEY_INDEX),
                originalSuggestions.get(OUTPUT_KEY_DATE_FIELDS),
                tenantId,
                maxRetries,
                currentRetry,
                "detector",
                listener,
                (suggestions, listenerCallback) -> validateDetectorPhase(suggestions, tenantId, maxRetries, currentRetry, listenerCallback)
            );
        }, e -> {
            log.error("Failed to get LLM fix for detector validation: " + e);
            listener.onFailure(e);
        }));
    }

    private AnomalyDetector applySuggestionsToDetector(AnomalyDetector originalDetector, SuggestConfigParamResponse response) {
        log.info("Applying suggestions to detector configuration");

        // Extract suggestions from response
        TimeConfiguration newInterval = originalDetector.getInterval();
        TimeConfiguration newWindowDelay = originalDetector.getWindowDelay();
        Integer newHistoryIntervals = originalDetector.getHistoryIntervals();

        // Apply interval suggestion if present
        if (response.getInterval() != null) {
            newInterval = response.getInterval();
            log.info("Applied suggested detection interval: {}", newInterval);
        }

        // Apply windowDelay suggestion if present
        if (response.getWindowDelay() != null) {
            newWindowDelay = response.getWindowDelay();
            log.info("Applied suggested window delay: {}", newWindowDelay);
        }

        // Apply history suggestion if present
        if (response.getHistory() != null) {
            newHistoryIntervals = response.getHistory();
            log.info("Applied suggested history intervals: {}", newHistoryIntervals);
        }

        // Create new detector with applied suggestions
        return new AnomalyDetector(
            originalDetector.getId(),
            originalDetector.getVersion(),
            originalDetector.getName(),
            originalDetector.getDescription(),
            originalDetector.getTimeField(),
            originalDetector.getIndices(),
            originalDetector.getFeatureAttributes(),
            originalDetector.getFilterQuery(),
            newInterval, // Updated
            newWindowDelay, // Updated
            originalDetector.getShingleSize(),
            originalDetector.getUiMetadata(),
            originalDetector.getSchemaVersion(),
            originalDetector.getLastUpdateTime(),
            originalDetector.getCategoryFields(),
            originalDetector.getUser(),
            originalDetector.getCustomResultIndexOrAlias(), // Fixed method name
            originalDetector.getImputationOption(),
            originalDetector.getRecencyEmphasis(),
            originalDetector.getSeasonIntervals(),
            newHistoryIntervals, // Updated
            originalDetector.getRules(),
            originalDetector.getCustomResultIndexMinSize(),
            originalDetector.getCustomResultIndexMinAge(),
            originalDetector.getCustomResultIndexTTL(),
            originalDetector.getFlattenResultIndexMapping(),
            null, // lastUIBreakingChangeTime - not available in getter, use null
            originalDetector.getFrequency(), // frequency
            originalDetector.getAutoCreated() // autoCreated - preserve original value
        );
    }

    private <T> void suggestHyperParametersPhase(AnomalyDetector detector, String tenantId, int maxRetries, ActionListener<T> listener) {
        log.info("Phase 2: Starting suggest hyper-parameters phase");

        // Create suggest request for interval, history, window_delay
        SuggestConfigParamRequest suggestRequest = new SuggestConfigParamRequest(
            AnalysisType.AD,
            detector,
            "interval,history,window_delay", // Request all parameters
            TimeValue.timeValueSeconds(30)
        );

        adClient.suggestAnomalyDetector(suggestRequest, new ActionListener<SuggestConfigParamResponse>() {
            @Override
            public void onResponse(SuggestConfigParamResponse response) {
                log.info("Phase 2: Suggest API succeeded");
                log.info("Phase 2: Suggest response: {}", response);

                try {
                    // Apply suggestions to detector (handles partial/empty responses gracefully)
                    AnomalyDetector optimizedDetector = applySuggestionsToDetector(detector, response);
                    log.info("Phase 2: Applied suggestions, proceeding to model validation");

                    // Proceed to Phase 3 - Model Validation (placeholder for now)
                    validateModelPhase(optimizedDetector, tenantId, maxRetries, 0, listener);

                } catch (Exception e) {
                    log.warn("Phase 2: Error applying suggestions, continuing with original detector: {}", e.getMessage());
                    // Continue with original detector if suggestion application fails
                    validateModelPhase(detector, tenantId, maxRetries, 0, listener);
                }
            }

            @Override
            public void onFailure(Exception e) {
                log.warn("Phase 2: Suggest API failed, continuing with original detector: {}", e.getMessage());
                // Continue to model validation with original detector
                validateModelPhase(detector, tenantId, maxRetries, 0, listener);
            }
        });
    }

    private <T> void validateModelPhase(
        AnomalyDetector detector,
        String tenantId,
        int maxRetries,
        int currentRetry,
        ActionListener<T> listener
    ) {
        log.info("Phase 3: Starting model validation phase");

        callValidationAPI(detector, "model", new ActionListener<ValidateConfigResponse>() {
            @Override
            public void onResponse(ValidateConfigResponse response) {
                log.info("Phase 3: Model validation API succeeded");
                log.info("Phase 3: Validation issue: {}", response.getIssue() == null ? "None" : response.getIssue().getMessage());

                if (response.getIssue() != null) {
                    String issueAspect = response.getIssue().getAspect().toString();
                    boolean isBlockingError = issueAspect != null && issueAspect.toLowerCase(Locale.ROOT).startsWith("detector");
                    log.info("Phase 3: Validation issue aspect: '{}', blocking: {}", issueAspect, isBlockingError);

                    String errorMessage = response.getIssue().getMessage();
                    if (currentRetry < MAX_MODEL_VALIDATION_RETRIES) {
                        retryModelValidation(detector, errorMessage, tenantId, maxRetries, currentRetry + 1, listener);
                    } else {
                        // Max retries reached
                        if (isBlockingError) {
                            log.error("Phase 3: Max retries reached with blocking validation error: {}", errorMessage);
                            DetectorResult result = DetectorResult.failedValidation(detector.getIndices().get(0), errorMessage);
                            listener.onResponse((T) result.toJson());
                        } else {
                            log.warn("Phase 3: Max retries reached with non-blocking validation warning: {}", errorMessage);
                            DetectorResult result = DetectorResult
                                .failedValidation(detector.getIndices().get(0), "Non-blocking warning: " + errorMessage);
                            listener.onResponse((T) result.toJson());
                        }
                    }
                    return;
                }

                // Model validation passed - CREATE AND START DETECTOR
                log.info("Phase 3: Model validation passed - creating and starting detector");
                createDetector(detector, listener);
            }

            @Override
            public void onFailure(Exception e) {
                log.error("Phase 3: Model validation API failed: {}", e.getMessage(), e);
                if (currentRetry < MAX_MODEL_VALIDATION_RETRIES) {
                    retryModelValidation(detector, e.getMessage(), tenantId, maxRetries, currentRetry + 1, listener);
                } else {
                    log.warn("Phase 3: Max retries reached, validation API failure");
                    DetectorResult result = DetectorResult.failedValidation(detector.getIndices().get(0), "API failure: " + e.getMessage());
                    listener.onResponse((T) result.toJson());
                }
            }
        });
    }

    private <T> void retryModelValidation(
        AnomalyDetector detector,
        String validationError,
        String tenantId,
        int maxRetries,
        int currentRetry,
        ActionListener<T> listener
    ) {
        log.info("Phase 3: Retrying model validation with LLM fix (attempt {}/3)", currentRetry);

        Map<String, String> currentSuggestions = Map
            .of(
                OUTPUT_KEY_INDEX,
                String.join(",", detector.getIndices()),
                OUTPUT_KEY_CATEGORY_FIELD,
                detector.getCategoryFields() == null || detector.getCategoryFields().isEmpty() ? "" : detector.getCategoryFields().get(0),
                OUTPUT_KEY_AGGREGATION_FIELD,
                detector.getFeatureAttributes().stream().map(f -> f.getName()).collect(java.util.stream.Collectors.joining(",")),
                OUTPUT_KEY_AGGREGATION_METHOD,
                detector.getFeatureAttributes().stream().map(f -> "avg").collect(java.util.stream.Collectors.joining(",")),
                OUTPUT_KEY_DATE_FIELDS,
                detector.getTimeField(),
                "interval",
                String.valueOf(detector.getIntervalInMinutes())
            );

        String fixPrompt = createFixPrompt(currentSuggestions, validationError);

        RemoteInferenceInputDataSet inputDataSet = RemoteInferenceInputDataSet
            .builder()
            .parameters(Collections.singletonMap("prompt", fixPrompt))
            .build();
        ActionRequest request = new MLPredictionTaskRequest(
            modelId,
            MLInput.builder().algorithm(FunctionName.REMOTE).inputDataset(inputDataSet).build(),
            null,
            tenantId
        );

        client.execute(MLPredictionTaskAction.INSTANCE, request, ActionListener.wrap(mlTaskResponse -> {
            ModelTensorOutput modelTensorOutput = (ModelTensorOutput) mlTaskResponse.getOutput();
            ModelTensors modelTensors = modelTensorOutput.getMlModelOutputs().get(0);
            ModelTensor modelTensor = modelTensors.getMlModelTensors().get(0);
            Map<String, Object> dataAsMap = (Map<String, Object>) modelTensor.getDataAsMap();
            String fixedResponse = extractResponseFromDataAsMap(dataAsMap);

            if (Strings.isNullOrEmpty(fixedResponse)) {
                log.warn("Phase 3: No LLM response for fix");
                DetectorResult result = DetectorResult.failedValidation(detector.getIndices().get(0), "LLM fix failed: no response");
                listener.onResponse((T) result.toJson());
                return;
            }

            parseAndRetryWithLLM(
                fixedResponse,
                currentSuggestions.get(OUTPUT_KEY_INDEX),
                currentSuggestions.get(OUTPUT_KEY_DATE_FIELDS),
                tenantId,
                maxRetries,
                currentRetry,
                "model",
                listener,
                (suggestions, listenerCallback) -> {
                    try {
                        AnomalyDetector newDetector = buildAnomalyDetectorFromSuggestions(suggestions);
                        validateModelPhase(newDetector, tenantId, maxRetries, currentRetry, listenerCallback);
                    } catch (Exception e) {
                        log.error("Phase 3: Error building detector from LLM fix: {}", e.getMessage());
                        DetectorResult result = DetectorResult
                            .failedValidation(detector.getIndices().get(0), "Failed to build detector: " + e.getMessage());
                        listenerCallback.onResponse((T) result.toJson());
                    }
                }
            );
        }, e -> {
            log.error("Phase 3: Failed to get LLM fix: " + e);
            DetectorResult result = DetectorResult
                .failedValidation(detector.getIndices().get(0), "LLM fix request failed: " + e.getMessage());
            listener.onResponse((T) result.toJson());
        }));
    }

    private <T> void createDetector(AnomalyDetector detector, ActionListener<T> listener) {
        IndexAnomalyDetectorRequest request = new IndexAnomalyDetectorRequest("", detector, RestRequest.Method.POST);

        log.info("Calling adClient.createAnomalyDetector for detector: {}", detector.getName());

        adClient.createAnomalyDetector(request, new ActionListener<org.opensearch.ad.transport.IndexAnomalyDetectorResponse>() {
            @Override
            public void onResponse(org.opensearch.ad.transport.IndexAnomalyDetectorResponse response) {
                String detectorId = response.getId();
                log.info("Created detector with ID: {}", detectorId);
                startDetector(detector.getIndices().get(0), detectorId, detector.getName(), listener);
            }

            @Override
            public void onFailure(Exception e) {
                log.error("Failed to create detector: {}", e.getMessage(), e);
                DetectorResult result = DetectorResult.failedCreate(detector.getIndices().get(0), e.getMessage());
                listener.onResponse((T) result.toJson());
            }
        });
    }

    private <T> void startDetector(String indexName, String detectorId, String detectorName, ActionListener<T> listener) {
        JobRequest request = new JobRequest(
            detectorId,
            ".opendistro-anomaly-detectors", // configIndex - the detector config index
            null, // dateRange - null for real-time detection
            false, // historical - false for real-time detection
            "/_plugins/_anomaly_detection/detectors/" + detectorId + "/_start" // rawPath - indicates start action
        );

        adClient.startAnomalyDetector(request, ActionListener.wrap(response -> {
            log.info("Started detector: {}", detectorId);
            DetectorResult result = DetectorResult
                .success(indexName, detectorId, detectorName, "Detector created successfully", "Detector started successfully");
            listener.onResponse((T) result.toJson());
        }, e -> {
            log.error("Failed to start detector: {}", e.getMessage());
            DetectorResult result = DetectorResult.failedStart(indexName, detectorId, e.getMessage());
            listener.onResponse((T) result.toJson());
        }));
    }

    private void selectOptimalDateField(String indexName, String[] suggestedDateFields, ActionListener<String> listener) {
        log.info("Selecting optimal date field from suggestions: {}", Arrays.toString(suggestedDateFields));

        // Query each date field to count recent documents (last 30 days)
        List<CompletableFuture<Pair<String, Long>>> futures = new ArrayList<>();

        for (String dateField : suggestedDateFields) {
            CompletableFuture<Pair<String, Long>> future = new CompletableFuture<>();
            futures.add(future);

            // Count documents with recent data for this date field
            SearchRequest searchRequest = new SearchRequest(indexName)
                .source(
                    new SearchSourceBuilder()
                        .query(
                            QueryBuilders
                                .boolQuery()
                                .must(QueryBuilders.existsQuery(dateField.trim()))
                                .must(QueryBuilders.rangeQuery(dateField.trim()).gte("now-30d").lte("now"))
                        )
                        .size(0)
                        .timeout(TimeValue.timeValueSeconds(10))
                );

            client.search(searchRequest, ActionListener.wrap((SearchResponse response) -> {
                long count = response.getHits().getTotalHits().value();
                log.info("Date field '{}' has {} documents with recent data (last 30 days)", dateField.trim(), count);
                future.complete(new Pair<>(dateField.trim(), count));
            }, e -> {
                log.warn("Failed to query date field '{}': {}", dateField.trim(), e.getMessage());
                future.complete(new Pair<>(dateField.trim(), 0L));
            }));
        }

        // Wait for all queries to complete and select the field with most recent data
        CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).thenAccept(v -> {
            String bestDateField = futures
                .stream()
                .map(CompletableFuture::join)
                .max(Comparator.comparing(Pair::getValue))
                .map(Pair::getKey)
                .orElse(suggestedDateFields[0]); // fallback to first suggestion

            log.info("Selected optimal date field: {} (best recent data count)", bestDateField);
            listener.onResponse(bestDateField);
        }).exceptionally(e -> {
            log.error("Error selecting optimal date field, using first suggestion: {}", e.getMessage());
            log.error("error1: ", e);
            listener.onResponse(suggestedDateFields[0]);
            return null;
        });
    }

    // Simple Pair class for holding field name and count
    private static class Pair<K, V> {
        private final K key;
        private final V value;

        public Pair(K key, V value) {
            this.key = key;
            this.value = value;
        }

        public K getKey() {
            return key;
        }

        public V getValue() {
            return value;
        }
    }

    private String createFixPrompt(Map<String, String> originalSuggestions, String validationError) {
        String currentInterval = originalSuggestions.getOrDefault("interval", "10");
        String categoryField = originalSuggestions.get(OUTPUT_KEY_CATEGORY_FIELD);

        // Check if this is a sparse data issue with unreasonably high suggested interval
        boolean isUnreasonableInterval = validationError.contains("interval")
            && (validationError.contains("240")
                || validationError.contains("480")
                || validationError.contains("960")
                || validationError.contains("1440"));

        String sparseDataGuidance = "";
        if (isUnreasonableInterval) {
            sparseDataGuidance = "\n**UNREASONABLE INTERVAL DETECTED**:\n"
                + "- Validation suggests interval >4 hours, which is impractical for most use cases\n"
                + "- PREFERRED SOLUTION: Remove category field entirely (set to empty)\n"
                + "- ALTERNATIVE: Choose different category field with lower cardinality\n"
                + "- Keep interval reasonable: 10-60 minutes for operational monitoring\n"
                + "- Better to have no segmentation than 8+ hour intervals\n";
        } else if (validationError.contains("sparse data") || validationError.contains("interval")) {
            sparseDataGuidance = "\n**SPARSE DATA GUIDANCE**:\n"
                + "- For intervals 60-120 min: acceptable, proceed with suggestion\n"
                + "- For intervals >120 min: consider removing category field instead\n";
        }

        return "VALIDATION ERROR: "
            + validationError
            + "\n\n"
            + "Current Configuration:\n"
            + "- Category Field: "
            + (categoryField == null || categoryField.isEmpty() ? "NONE" : categoryField)
            + "\n"
            + "- Aggregation Fields: "
            + originalSuggestions.get(OUTPUT_KEY_AGGREGATION_FIELD)
            + "\n"
            + "- Aggregation Methods: "
            + originalSuggestions.get(OUTPUT_KEY_AGGREGATION_METHOD)
            + "\n"
            + "- Interval: "
            + currentInterval
            + " minutes\n"
            + sparseDataGuidance
            + "\n\nFIX STRATEGY:\n"
            + "1. For intervals >240 min suggested: REMOVE category field (set to empty string)\n"
            + "2. For intervals 60-120 min: accept the suggested interval\n"
            + "3. For intervals 15-60 min: accept the suggested interval\n"
            + "4. For 'invalid query' errors: fix only the problematic field/method\n"
            + "5. Prefer operational usefulness over perfect validation\n\n"
            + "CRITICAL: Return ONLY the corrected configuration in this EXACT format:\n"
            + "{category_field=FIELD_OR_EMPTY|aggregation_field=FIELD1,FIELD2|aggregation_method=METHOD1,METHOD2|interval=MINUTES}\n\n"
            + "Use empty string for category_field if removing it. DO NOT include explanations.";
    }

    // ==================== EXTRACTED HELPER METHODS ====================

    /**
     * Context object to hold mapping data
     */
    private static class MappingContext {
        final String indexName;
        final Map<String, String> filteredMapping;
        final Set<String> dateFields;

        MappingContext(String indexName, Map<String, String> filteredMapping, Set<String> dateFields) {
            this.indexName = indexName;
            this.filteredMapping = filteredMapping;
            this.dateFields = dateFields;
        }
    }

    /**
     * Result object to track detector creation status per index
     */
    private static class DetectorResult {
        String indexName;
        String status; // "success", "failed_validation", "failed_create", "failed_start"
        String detectorId;
        String detectorName;
        String error;
        String createResponse;
        String startResponse;

        String toJson() {
            Map<String, String> map = new HashMap<>();
            if (indexName != null)
                map.put("indexName", indexName);
            if (status != null)
                map.put("status", status);
            if (detectorId != null)
                map.put("detectorId", detectorId);
            if (detectorName != null)
                map.put("detectorName", detectorName);
            if (error != null)
                map.put("error", error);
            if (createResponse != null)
                map.put("createResponse", createResponse);
            if (startResponse != null)
                map.put("startResponse", startResponse);
            return gson.toJson(map);
        }

        static DetectorResult failedValidation(String indexName, String error) {
            DetectorResult result = new DetectorResult();
            result.indexName = indexName;
            result.status = "failed_validation";
            result.error = error;
            return result;
        }

        static DetectorResult failedCreate(String indexName, String error) {
            DetectorResult result = new DetectorResult();
            result.indexName = indexName;
            result.status = "failed_create";
            result.error = error;
            return result;
        }

        static DetectorResult failedStart(String indexName, String detectorId, String error) {
            DetectorResult result = new DetectorResult();
            result.indexName = indexName;
            result.status = "failed_start";
            result.detectorId = detectorId;
            result.error = error;
            return result;
        }

        static DetectorResult success(
            String indexName,
            String detectorId,
            String detectorName,
            String createResponse,
            String startResponse
        ) {
            DetectorResult result = new DetectorResult();
            result.indexName = indexName;
            result.status = "success";
            result.detectorId = detectorId;
            result.detectorName = detectorName;
            result.createResponse = createResponse;
            result.startResponse = startResponse;
            return result;
        }
    }

    /**
     * Step 1: Extract and validate index name from parameters
     */
    private String extractAndValidateIndexName(Map<String, String> parameters) {
        Map<String, String> enrichedParameters = enrichParameters(parameters);
        String indexName = enrichedParameters.get("index");

        if (Strings.isNullOrEmpty(indexName)) {
            throw new IllegalArgumentException(
                "Return this final answer to human directly and do not use other tools: 'Please provide index name'. "
                    + "Please try to directly send this message to human to ask for index name"
            );
        }

        if (indexName.startsWith(".")) {
            throw new IllegalArgumentException(
                "CreateAnomalyDetectionTool doesn't support searching indices starting with '.' since it could be system index, "
                    + "current searching index name: "
                    + indexName
            );
        }

        return indexName;
    }

    /**
     * Step 2: Get mappings and filter fields
     */
    private void getMappingsAndFilterFields(String indexName, ActionListener<MappingContext> listener) {
        GetMappingsRequest getMappingsRequest = new GetMappingsRequest().indices(indexName);

        client.admin().indices().getMappings(getMappingsRequest, ActionListener.wrap(response -> {
            Map<String, MappingMetadata> mappings = response.getMappings();

            if (mappings.size() == 0) {
                listener.onFailure(new IllegalArgumentException("No mapping found for the index: " + indexName));
                return;
            }

            String firstIndexName = (String) mappings.keySet().toArray()[0];
            MappingMetadata mappingMetadata = mappings.get(firstIndexName);
            Map<String, Object> mappingSource = (Map<String, Object>) mappingMetadata.getSourceAsMap().get("properties");

            if (Objects.isNull(mappingSource)) {
                listener
                    .onFailure(
                        new IllegalArgumentException(
                            "The index " + indexName + " doesn't have mapping metadata, please add data to it or using another index."
                        )
                    );
                return;
            }

            Map<String, String> fieldsToType = new HashMap<>();
            ToolHelper.extractFieldNamesTypes(mappingSource, fieldsToType, "", true);

            final Set<String> dateFields = findDateTypeFields(fieldsToType);
            if (dateFields.isEmpty()) {
                listener
                    .onFailure(
                        new IllegalArgumentException(
                            "The index " + indexName + " doesn't have date type fields, cannot create an anomaly detector for it."
                        )
                    );
                return;
            }

            Map<String, String> filteredMapping = fieldsToType
                .entrySet()
                .stream()
                .filter(entry -> VALID_FIELD_TYPES.contains(entry.getValue()))
                .collect(Collectors.toUnmodifiableMap(Map.Entry::getKey, Map.Entry::getValue));

            MappingContext context = new MappingContext(firstIndexName, filteredMapping, dateFields);
            listener.onResponse(context);

        }, e -> {
            log.error("failed to get mapping: " + e);
            if (e.toString().contains("IndexNotFoundException")) {
                listener
                    .onFailure(
                        new IllegalArgumentException(
                            "Return this final answer to human directly and do not use other tools: "
                                + "'The index doesn't exist, please provide another index and retry'. "
                                + "Please try to directly send this message to human to ask for index name"
                        )
                    );
            } else {
                listener.onFailure(e);
            }
        }));
    }

    /**
     * Step 3-6: Generate config with LLM, parse, and select date field
     */
    private <T> void generateAndParseConfig(
        MappingContext mappingContext,
        String tenantId,
        int maxRetries,
        ActionListener<T> listener,
        java.util.function.BiConsumer<Map<String, String>, ActionListener<T>> nextPhaseCallback
    ) {
        StringJoiner dateFieldsJoiner = new StringJoiner(",");
        mappingContext.dateFields.forEach(dateFieldsJoiner::add);

        String prompt = constructPrompt(mappingContext.filteredMapping, mappingContext.indexName, dateFieldsJoiner.toString());

        RemoteInferenceInputDataSet inputDataSet = RemoteInferenceInputDataSet
            .builder()
            .parameters(
                Map
                    .of(
                        "prompt",
                        prompt,
                        "system_prompt",
                        "You are an expert in OpenSearch anomaly detection. "
                            + "Analyze the provided index mapping and suggest appropriate anomaly detector configurations."
                    )
            )
            .build();

        ActionRequest request = new MLPredictionTaskRequest(
            modelId,
            MLInput.builder().algorithm(FunctionName.REMOTE).inputDataset(inputDataSet).build(),
            null,
            tenantId
        );

        client.execute(MLPredictionTaskAction.INSTANCE, request, ActionListener.wrap(mlTaskResponse -> {
            ModelTensorOutput modelTensorOutput = (ModelTensorOutput) mlTaskResponse.getOutput();
            ModelTensors modelTensors = modelTensorOutput.getMlModelOutputs().get(0);
            ModelTensor modelTensor = modelTensors.getMlModelTensors().get(0);
            Map<String, Object> dataAsMap = (Map<String, Object>) modelTensor.getDataAsMap();

            if (dataAsMap == null) {
                listener.onFailure(new IllegalStateException("Remote endpoint fails to inference."));
                return;
            }

            String finalResponse = extractResponseFromDataAsMap(dataAsMap);
            if (Strings.isNullOrEmpty(finalResponse)) {
                listener.onFailure(new IllegalStateException("Remote endpoint fails to inference, no response found."));
                return;
            }

            // Parse response and continue to next phase
            parseAndRetryWithLLM(
                finalResponse,
                mappingContext.indexName,
                dateFieldsJoiner.toString(),
                tenantId,
                maxRetries,
                0,
                "model",
                listener,
                nextPhaseCallback
            );

        }, e -> {
            log.error("fail to predict model: " + e);
            listener.onFailure(e);
        }));
    }

    /**
     * The tool factory
     */
    public static class Factory implements WithModelTool.Factory<CreateAnomalyDetectorToolEnhanced> {
        private Client client;
        private NamedWriteableRegistry namedWriteableRegistry;

        private static CreateAnomalyDetectorToolEnhanced.Factory INSTANCE;

        /**
         * Create or return the singleton factory instance
         */
        public static CreateAnomalyDetectorToolEnhanced.Factory getInstance() {
            if (INSTANCE != null) {
                return INSTANCE;
            }
            synchronized (CreateAnomalyDetectorToolEnhanced.class) {
                if (INSTANCE != null) {
                    return INSTANCE;
                }
                INSTANCE = new CreateAnomalyDetectorToolEnhanced.Factory();
                return INSTANCE;
            }
        }

        public void init(Client client, NamedWriteableRegistry namedWriteableRegistry) {
            this.client = client;
            this.namedWriteableRegistry = namedWriteableRegistry;
        }

        /**
         *
         * @param map the input parameters
         * @return the instance of this tool
         */
        @Override
        public CreateAnomalyDetectorToolEnhanced create(Map<String, Object> map) {
            String modelId = (String) map.getOrDefault(COMMON_MODEL_ID_FIELD, "");
            if (modelId.isEmpty()) {
                throw new IllegalArgumentException("model_id cannot be empty.");
            }
            String modelType = (String) map.getOrDefault("model_type", ModelType.CLAUDE.toString());
            // if model type is empty, use the default value
            if (modelType.isEmpty()) {
                modelType = ModelType.CLAUDE.toString();
            } else if (!ModelType.OPENAI.toString().equalsIgnoreCase(modelType)
                && !ModelType.CLAUDE.toString().equalsIgnoreCase(modelType)) {
                throw new IllegalArgumentException("Unsupported model_type: " + modelType);
            }
            String prompt = (String) map.getOrDefault("prompt", "");
            String promptFile = (String) map.getOrDefault("prompt_file", "");
            return new CreateAnomalyDetectorToolEnhanced(client, modelId, modelType, prompt, promptFile, namedWriteableRegistry);
        }

        @Override
        public String getDefaultDescription() {
            return DEFAULT_DESCRIPTION;
        }

        @Override
        public String getDefaultType() {
            return TYPE;
        }

        @Override
        public String getDefaultVersion() {
            return null;
        }

        @Override
        public List<String> getAllModelKeys() {
            return List.of(COMMON_MODEL_ID_FIELD);
        }
    }
}
