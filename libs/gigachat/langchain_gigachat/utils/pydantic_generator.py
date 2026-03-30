from typing import Union

from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue, core_schema


class GigaChatJsonSchema(GenerateJsonSchema):
    def field_is_required(
        self,
        field: Union[
            core_schema.ModelField,
            core_schema.DataclassField,
            core_schema.TypedDictField,
        ],
        total: bool,
    ) -> bool:
        """
        Make nullable fields not required
        """
        if field["schema"]["type"] == "nullable":
            return False
        return super().field_is_required(field, total)

    def nullable_schema(self, schema: core_schema.NullableSchema) -> JsonSchemaValue:
        """
        Remove anyOf if field is nullable
        """
        null_schema = {"type": "null"}
        inner_json_schema = self.generate_inner(schema["schema"])

        if inner_json_schema == null_schema:
            return null_schema
        else:
            return inner_json_schema

    def model_schema(self, schema: core_schema.ModelSchema) -> JsonSchemaValue:
        """
        Add ``"default": null`` for nullable fields that have no explicit default.

        GigaChat omits Optional fields from tool-call responses instead of
        returning ``null``.  Without a ``"default": null`` hint in the schema,
        PydanticToolsParser raises ``ValidationError: Field required`` when such
        a field is missing from the response.
        """
        result = super().model_schema(schema)
        if isinstance(result, dict) and "properties" in result:
            required = result.get("required", [])
            for prop_name, prop_schema in result["properties"].items():
                if (
                    isinstance(prop_schema, dict)
                    and prop_name not in required
                    and "default" not in prop_schema
                ):
                    prop_schema["default"] = None
        return result
