"""Workflow Run RO-Crate construction using the reference Python library."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rocrate.model.computerlanguage import ComputerLanguage
from rocrate.model.contextentity import ContextEntity
from rocrate.rocrate import ROCrate

from .io import sha256_file

WORKFLOW_RUN_CONTEXT = "https://w3id.org/ro/terms/workflow-run/context"
PROFILES = (
    ("https://w3id.org/ro/wfrun/workflow/0.5", "Workflow Run Crate", "0.5"),
    ("https://w3id.org/ro/wfrun/process/0.5", "Process Run Crate", "0.5"),
    ("https://w3id.org/workflowhub/workflow-ro-crate/1.0", "Workflow RO-Crate", "1.0"),
)


def write_workflow_run_crate(
    output: Path,
    root: Path,
    provenance: dict[str, Any],
    assessment: dict[str, Any],
) -> None:
    """Describe one completed attempt as a Workflow Run RO-Crate 0.5."""
    crate = ROCrate(version="1.1")
    crate.metadata.extra_contexts.append(WORKFLOW_RUN_CONTEXT)

    language = crate.add(
        ComputerLanguage(
            crate,
            "#python",
            properties={
                "name": "Python",
                "version": provenance["environment"]["python"],
                "url": {"@id": "https://www.python.org/"},
            },
        )
    )
    workflow = crate.add_workflow(
        output / "workflow.json",
        "workflow.json",
        main=True,
        lang=language,
        properties={
            "name": "src-case run",
            "description": "Registered paired MNIST/QMNIST analysis entry point",
            "encodingFormat": "application/json",
            "sha256": sha256_file(output / "workflow.json"),
        },
    )

    files = {}
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name not in {"ro-crate-metadata.json", "workflow.json"}:
            files[path.name] = crate.add_file(
                path,
                path.name,
                properties={
                    "name": path.name,
                    "contentSize": str(path.stat().st_size),
                    "sha256": sha256_file(path),
                },
            )

    inputs = []
    for record in provenance["external_inputs"]:
        properties = {"name": record.get("name", record["role"])}
        if "url" in record:
            properties.update({"contentSize": str(record["bytes"]), "sha256": record["sha256"]})
            entity = crate.add_file(record["url"], properties=properties)
        else:
            source = root / record["path"]
            properties.update(
                {
                    "contentSize": str(source.stat().st_size),
                    "sha256": sha256_file(source),
                    "identifier": f"urn:sha256:{record['payload_sha256']}",
                }
            )
            entity = crate.add_file(
                source,
                f"inputs/{source.name}",
                properties=properties,
            )
        inputs.append(entity)

    result_names = {record["path"] for record in provenance["outputs"]}
    action = crate.add_action(
        workflow,
        "#attempt",
        object=inputs,
        result=[files[name] for name in sorted(result_names)],
        properties={
            "name": provenance["attempt_id"],
            "description": assessment["decision"] or assessment["classification"],
            "startTime": provenance["started_at"],
            "endTime": provenance["ended_at"],
            "actionStatus": {
                "@id": "http://schema.org/CompletedActionStatus"
                if provenance["termination"]["status"] == "completed"
                else "http://schema.org/FailedActionStatus"
            },
        },
    )
    for name, entity in files.items():
        if name not in result_names:
            entity["about"] = action

    profiles = [
        crate.add(
            ContextEntity(
                crate,
                identifier,
                properties={"@type": "CreativeWork", "name": name, "version": version},
            )
        )
        for identifier, name, version in PROFILES
    ]
    crate.root_dataset["name"] = f"SRC assessment {provenance['attempt_id']}"
    crate.root_dataset["description"] = "Execution evidence for one registered SRC attempt"
    crate.root_dataset["datePublished"] = provenance["ended_at"]
    crate.root_dataset["license"] = {"@id": "https://spdx.org/licenses/Apache-2.0"}
    crate.root_dataset["conformsTo"] = profiles
    crate.root_dataset["mentions"] = action
    crate.write(output)
