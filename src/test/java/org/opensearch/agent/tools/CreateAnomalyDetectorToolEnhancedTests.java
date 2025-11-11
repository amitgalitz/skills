/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.agent.tools;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.when;
import static org.opensearch.ml.common.utils.StringUtils.gson;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import org.junit.Before;
import org.junit.Test;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.opensearch.action.admin.indices.mapping.get.GetMappingsResponse;
import org.opensearch.cluster.metadata.MappingMetadata;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.common.io.stream.NamedWriteableRegistry;
import org.opensearch.ml.common.output.model.MLResultDataType;
import org.opensearch.ml.common.output.model.ModelTensor;
import org.opensearch.ml.common.output.model.ModelTensorOutput;
import org.opensearch.ml.common.output.model.ModelTensors;
import org.opensearch.ml.common.transport.MLTaskResponse;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskAction;
import org.opensearch.transport.client.AdminClient;
import org.opensearch.transport.client.Client;
import org.opensearch.transport.client.IndicesAdminClient;

import com.google.common.collect.ImmutableMap;

import lombok.extern.log4j.Log4j2;

@Log4j2
public class CreateAnomalyDetectorToolEnhancedTests {
    @Mock
    private Client client;
    @Mock
    private AdminClient adminClient;
    @Mock
    private IndicesAdminClient indicesAdminClient;
    @Mock
    private GetMappingsResponse getMappingsResponse;
    @Mock
    private MappingMetadata mappingMetadata;
    @Mock
    private NamedWriteableRegistry namedWriteableRegistry;

    private Map<String, MappingMetadata> mockedMappings;
    private Map<String, Object> indexMappings;

    @Mock
    private MLTaskResponse mlTaskResponse;
    @Mock
    private ModelTensorOutput modelTensorOutput;
    @Mock
    private ModelTensors modelTensors;

    private ModelTensor modelTensor;
    private Map<String, ?> modelReturns;

    private String mockedIndexName = "http_logs";
    private String mockedResponse = "{category_field=|aggregation_field=response,responseLatency|aggregation_method=count,avg|interval=10}";

    @Before
    public void setup() {
        MockitoAnnotations.openMocks(this);
        createMappings();

        // Setup mapping mocks
        when(mappingMetadata.getSourceAsMap()).thenReturn(indexMappings);
        when(getMappingsResponse.getMappings()).thenReturn(mockedMappings);
        when(client.admin()).thenReturn(adminClient);
        when(adminClient.indices()).thenReturn(indicesAdminClient);
        doAnswer(invocation -> {
            ActionListener<GetMappingsResponse> listener = (ActionListener<GetMappingsResponse>) invocation.getArguments()[1];
            listener.onResponse(getMappingsResponse);
            return null;
        }).when(indicesAdminClient).getMappings(any(), any());

        initMLTensors();
        CreateAnomalyDetectorToolEnhanced.Factory.getInstance().init(client, namedWriteableRegistry);
    }

    @Test
    public void testModelIdIsNullOrEmpty() {
        Exception exception = assertThrows(
            IllegalArgumentException.class,
            () -> CreateAnomalyDetectorToolEnhanced.Factory.getInstance().create(ImmutableMap.of("model_id", ""))
        );
        assertEquals("model_id cannot be empty.", exception.getMessage());
    }

    @Test
    public void testInvalidModelType() {
        Exception exception = assertThrows(
            IllegalArgumentException.class,
            () -> CreateAnomalyDetectorToolEnhanced.Factory
                .getInstance()
                .create(ImmutableMap.of("model_id", "modelId", "model_type", "unknown"))
        );
        assertEquals("Unsupported model_type: unknown", exception.getMessage());
    }

    @Test
    public void testValidModelTypes() {
        CreateAnomalyDetectorToolEnhanced tool = CreateAnomalyDetectorToolEnhanced.Factory
            .getInstance()
            .create(ImmutableMap.of("model_id", "modelId", "model_type", "openai"));
        assertEquals(CreateAnomalyDetectorToolEnhanced.TYPE, tool.getName());
        assertEquals("modelId", tool.getModelId());
        assertEquals("OPENAI", tool.getModelType().toString());

        tool = CreateAnomalyDetectorToolEnhanced.Factory
            .getInstance()
            .create(ImmutableMap.of("model_id", "modelId", "model_type", "claude"));
        assertEquals(CreateAnomalyDetectorToolEnhanced.TYPE, tool.getName());
        assertEquals("modelId", tool.getModelId());
        assertEquals("CLAUDE", tool.getModelType().toString());
    }

    @Test
    public void testDefaultModelType() {
        CreateAnomalyDetectorToolEnhanced tool = CreateAnomalyDetectorToolEnhanced.Factory
            .getInstance()
            .create(ImmutableMap.of("model_id", "modelId"));
        assertEquals("CLAUDE", tool.getModelType().toString());
    }

    @Test
    public void testEmptyModelType() {
        CreateAnomalyDetectorToolEnhanced tool = CreateAnomalyDetectorToolEnhanced.Factory
            .getInstance()
            .create(ImmutableMap.of("model_id", "modelId", "model_type", ""));
        assertEquals("CLAUDE", tool.getModelType().toString());
    }

    @Test
    public void testCustomPrompt() {
        CreateAnomalyDetectorToolEnhanced tool = CreateAnomalyDetectorToolEnhanced.Factory
            .getInstance()
            .create(ImmutableMap.of("model_id", "modelId", "prompt", "custom prompt"));
        assertEquals("custom prompt", tool.getContextPrompt());
    }

    @Test
    public void testIndexNameValidation_Empty() {
        CreateAnomalyDetectorToolEnhanced tool = CreateAnomalyDetectorToolEnhanced.Factory
            .getInstance()
            .create(ImmutableMap.of("model_id", "modelId"));

        Exception exception = assertThrows(
            IllegalArgumentException.class,
            () -> tool
                .run(
                    ImmutableMap.of("input", gson.toJson(ImmutableMap.of("indices", Collections.emptyList()))),
                    ActionListener.<String>wrap(response -> {}, e -> {
                        throw new IllegalArgumentException(e.getMessage());
                    })
                )
        );
        assertTrue(exception.getMessage().contains("Please provide index name"));
    }

    @Test
    public void testIndexNameValidation_SystemIndex() {
        CreateAnomalyDetectorToolEnhanced tool = CreateAnomalyDetectorToolEnhanced.Factory
            .getInstance()
            .create(ImmutableMap.of("model_id", "modelId"));

        Exception exception = assertThrows(
            IllegalArgumentException.class,
            () -> tool
                .run(
                    ImmutableMap.of("input", gson.toJson(ImmutableMap.of("indices", Collections.singletonList(".system_index")))),
                    ActionListener.<String>wrap(response -> {}, e -> {
                        throw new IllegalArgumentException(e.getMessage());
                    })
                )
        );
        assertTrue(exception.getMessage().contains("doesn't support searching indices starting with '.'"));
    }

    @Test
    public void testInputFormat_IndicesList() {
        CreateAnomalyDetectorToolEnhanced tool = CreateAnomalyDetectorToolEnhanced.Factory
            .getInstance()
            .create(ImmutableMap.of("model_id", "modelId"));

        // This should extract first index from list
        tool
            .run(
                ImmutableMap.of("input", gson.toJson(ImmutableMap.of("indices", Collections.singletonList(mockedIndexName)))),
                ActionListener.<String>wrap(response -> {
                    log.info("Response: {}", response);
                    // Response should be DetectorResult JSON
                    assertTrue(response.contains("indexName"));
                }, e -> log.error("Error: ", e))
            );
    }

    @Test
    public void testLLMResponseParsing_ValidFormat() {
        CreateAnomalyDetectorToolEnhanced tool = CreateAnomalyDetectorToolEnhanced.Factory
            .getInstance()
            .create(ImmutableMap.of("model_id", "modelId"));

        tool
            .run(
                ImmutableMap.of("input", gson.toJson(ImmutableMap.of("indices", Collections.singletonList(mockedIndexName)))),
                ActionListener.<String>wrap(response -> {
                    log.info("Valid format response: {}", response);
                    // Should successfully parse and proceed through phases
                    assertTrue(response.contains("indexName"));
                }, e -> log.error("Error: ", e))
            );
    }

    @Test
    public void testLLMResponseParsing_InvalidFormat() {
        CreateAnomalyDetectorToolEnhanced tool = CreateAnomalyDetectorToolEnhanced.Factory
            .getInstance()
            .create(ImmutableMap.of("model_id", "modelId"));

        // Mock invalid response format
        modelReturns = Collections.singletonMap("response", "invalid format without curly braces");
        modelTensor = new ModelTensor("tensor", new Number[0], new long[0], MLResultDataType.STRING, null, null, modelReturns);
        initMLTensors();

        tool
            .run(
                ImmutableMap.of("input", gson.toJson(ImmutableMap.of("indices", Collections.singletonList(mockedIndexName)))),
                ActionListener.<String>wrap(response -> log.info("Response: {}", response), e -> {
                    log.info("Expected failure: {}", e.getMessage());
                    assertTrue(e.getMessage().contains("Cannot parse LLM response"));
                })
            );
    }

    @Test
    public void testLLMResponseParsing_EmptyResponse() {
        CreateAnomalyDetectorToolEnhanced tool = CreateAnomalyDetectorToolEnhanced.Factory
            .getInstance()
            .create(ImmutableMap.of("model_id", "modelId"));

        modelReturns = Collections.singletonMap("response", "");
        modelTensor = new ModelTensor("tensor", new Number[0], new long[0], MLResultDataType.STRING, null, null, modelReturns);
        initMLTensors();

        Exception exception = assertThrows(
            IllegalStateException.class,
            () -> tool
                .run(
                    ImmutableMap.of("input", gson.toJson(ImmutableMap.of("indices", Collections.singletonList(mockedIndexName)))),
                    ActionListener.<String>wrap(response -> {}, e -> {
                        throw new IllegalStateException(e.getMessage());
                    })
                )
        );
        assertTrue(exception.getMessage().contains("Remote endpoint fails to inference"));
    }

    @Test
    public void testLLMResponseParsing_NullResponse() {
        CreateAnomalyDetectorToolEnhanced tool = CreateAnomalyDetectorToolEnhanced.Factory
            .getInstance()
            .create(ImmutableMap.of("model_id", "modelId"));

        modelReturns = Collections.singletonMap("response", null);
        modelTensor = new ModelTensor("tensor", new Number[0], new long[0], MLResultDataType.STRING, null, null, modelReturns);
        initMLTensors();

        Exception exception = assertThrows(
            IllegalStateException.class,
            () -> tool
                .run(
                    ImmutableMap.of("input", gson.toJson(ImmutableMap.of("indices", Collections.singletonList(mockedIndexName)))),
                    ActionListener.<String>wrap(response -> {}, e -> {
                        throw new IllegalStateException(e.getMessage());
                    })
                )
        );
        assertTrue(exception.getMessage().contains("Remote endpoint fails to inference"));
    }

    @Test
    public void testIntervalParsing_ValidInterval() {
        // Test that interval is correctly parsed from LLM response
        String responseWithInterval = "{category_field=|aggregation_field=response|aggregation_method=count|interval=15}";
        modelReturns = Collections.singletonMap("response", responseWithInterval);
        modelTensor = new ModelTensor("tensor", new Number[0], new long[0], MLResultDataType.STRING, null, null, modelReturns);
        initMLTensors();

        CreateAnomalyDetectorToolEnhanced tool = CreateAnomalyDetectorToolEnhanced.Factory
            .getInstance()
            .create(ImmutableMap.of("model_id", "modelId"));

        tool
            .run(
                ImmutableMap.of("input", gson.toJson(ImmutableMap.of("indices", Collections.singletonList(mockedIndexName)))),
                ActionListener.<String>wrap(response -> log.info("Interval test response: {}", response), e -> log.error("Error: ", e))
            );
    }

    @Test
    public void testIntervalParsing_DefaultInterval() {
        // Test that default interval (10) is used when not specified
        String responseWithoutInterval = "{category_field=|aggregation_field=response|aggregation_method=count|interval=}";
        modelReturns = Collections.singletonMap("response", responseWithoutInterval);
        modelTensor = new ModelTensor("tensor", new Number[0], new long[0], MLResultDataType.STRING, null, null, modelReturns);
        initMLTensors();

        CreateAnomalyDetectorToolEnhanced tool = CreateAnomalyDetectorToolEnhanced.Factory
            .getInstance()
            .create(ImmutableMap.of("model_id", "modelId"));

        tool
            .run(
                ImmutableMap.of("input", gson.toJson(ImmutableMap.of("indices", Collections.singletonList(mockedIndexName)))),
                ActionListener
                    .<String>wrap(response -> log.info("Default interval test response: {}", response), e -> log.error("Error: ", e))
            );
    }

    @Test
    public void testCategoryField_Empty() {
        // Test single-entity detector (no category field)
        String responseNoCategoryField = "{category_field=|aggregation_field=response|aggregation_method=count|interval=10}";
        modelReturns = Collections.singletonMap("response", responseNoCategoryField);
        modelTensor = new ModelTensor("tensor", new Number[0], new long[0], MLResultDataType.STRING, null, null, modelReturns);
        initMLTensors();

        CreateAnomalyDetectorToolEnhanced tool = CreateAnomalyDetectorToolEnhanced.Factory
            .getInstance()
            .create(ImmutableMap.of("model_id", "modelId"));

        tool
            .run(
                ImmutableMap.of("input", gson.toJson(ImmutableMap.of("indices", Collections.singletonList(mockedIndexName)))),
                ActionListener.<String>wrap(response -> log.info("No category field response: {}", response), e -> log.error("Error: ", e))
            );
    }

    @Test
    public void testCategoryField_WithValue() {
        // Test multi-entity detector (with category field)
        String responseWithCategoryField = "{category_field=host|aggregation_field=response|aggregation_method=count|interval=10}";
        modelReturns = Collections.singletonMap("response", responseWithCategoryField);
        modelTensor = new ModelTensor("tensor", new Number[0], new long[0], MLResultDataType.STRING, null, null, modelReturns);
        initMLTensors();

        CreateAnomalyDetectorToolEnhanced tool = CreateAnomalyDetectorToolEnhanced.Factory
            .getInstance()
            .create(ImmutableMap.of("model_id", "modelId"));

        tool
            .run(
                ImmutableMap.of("input", gson.toJson(ImmutableMap.of("indices", Collections.singletonList(mockedIndexName)))),
                ActionListener
                    .<String>wrap(response -> log.info("With category field response: {}", response), e -> log.error("Error: ", e))
            );
    }

    private void createMappings() {
        indexMappings = new HashMap<>();
        indexMappings
            .put(
                "properties",
                ImmutableMap
                    .of(
                        "response",
                        ImmutableMap.of("type", "integer"),
                        "responseLatency",
                        ImmutableMap.of("type", "float"),
                        "host",
                        ImmutableMap.of("type", "keyword"),
                        "date",
                        ImmutableMap.of("type", "date")
                    )
            );
        mockedMappings = new HashMap<>();
        mockedMappings.put(mockedIndexName, mappingMetadata);

        modelReturns = Collections.singletonMap("response", mockedResponse);
        modelTensor = new ModelTensor("tensor", new Number[0], new long[0], MLResultDataType.STRING, null, null, modelReturns);
    }

    private void initMLTensors() {
        when(modelTensors.getMlModelTensors()).thenReturn(Collections.singletonList(modelTensor));
        when(modelTensorOutput.getMlModelOutputs()).thenReturn(Collections.singletonList(modelTensors));
        when(mlTaskResponse.getOutput()).thenReturn(modelTensorOutput);

        doAnswer(invocation -> {
            ActionListener<MLTaskResponse> listener = (ActionListener<MLTaskResponse>) invocation.getArguments()[2];
            listener.onResponse(mlTaskResponse);
            return null;
        }).when(client).execute(eq(MLPredictionTaskAction.INSTANCE), any(), any());
    }
}
