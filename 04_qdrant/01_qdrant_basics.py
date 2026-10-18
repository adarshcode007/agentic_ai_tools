import os
import shutil
import numpy as np 
from qdrant_client import QdrantClient, models

def main():
    storage_path = "./qdrant_storage"

    # Clean previous run data for a clean start
    if os.path.exists(storage_path):
        shutil.rmtree(storage_path)
    
    print("=" * 65)
    print("PART 1: INITIALIZE LOCAL EMBEDDED QDRANT CLIENT")
    print("=" * 65)

    # In local mode, Qdrant runs embedded in Python with disk persistence (no Docker needed!)
    client = QdrantClient(path=storage_path)

    collection_v1 = "products_v1"
    collection_v2 = "products_v2"
    d = 64    # Dimension

    # Create collection v1
    client.create_collection(
        collection_name=collection_v1,
        vectors_config=models.VectorParams(
            size=d,
            distance=models.Distance.COSINE
        )
    )
    print(f"Collection '{collection_v1}' created with size={d} and cosine distance.")

    # PART 2: INGEST POINTS WITH RICH JSON PAYLOADS

    print("\n" + "=" * 65)
    print("PART 2: INGESTING POINTSTRUCT OBJECTS")
    print("=" * 65)

    np.random.seed(42)

    # Generate synthetic embeddings
    vectors = [np.random.random(d).astype(np.float32).tolist() for _ in range(4)]

    # In Qdrant, every entity is a PointStruct: ID + Vector + Payload
    points_v1 = [
        models.PointStruct(
            id=1,
            vector=vectors[0],
            payload={"name": "Cloud Server Pro", "category": "compute", "price": 120.0, "active": True}
        ),
        models.PointStruct(
            id=2,
            vector=vectors[1],
            payload={"name": "Object Storage Glacier", "category": "storage", "price": 15.0, "active": True}
        ),
        models.PointStruct(
            id=3,
            vector=vectors[2],
            payload={"name": "Serverless Database", "category": "database", "price": 85.0, "active": False}
        ),
        models.PointStruct(
            id=4,
            vector=vectors[3],
            payload={"name": "High-Mem Cloud Server", "category": "compute", "price": 250.0, "active": True}
        )
    ]

    client.upsert(collection_name=collection_v1, points=points_v1)
    print(f"Upserted {len(points_v1)} points into '{collection_v1}'.")


    # PART 3: CREATE PAYLOAD INDEX (Optimizing Filtered HNSW)

    print("\n" + "=" * 65)
    print("PART 3: CREATING PAYLOAD INDEX FOR FILTERING")
    print("=" * 65)

    # Creating a payload index builds an inverted index on disk for that field
    # enabling Qdrant's single-stage filtered HNSW traversal.
    client.create_payload_index(
        collection_name=collection_v1,
        field_name="category",
        field_schema=models.PayloadSchemaType.KEYWORD
    )
    print("Payload index created on field: 'category' (KEYWORD)")


    # PART 4: SINGLE-STAGE FILTERED VECTOR SEARCH
    print("\n" + "=" * 65)
    print("PART 4: FILTERED SEARCH (category == 'compute' AND active == True)")
    print("=" * 65)

    query_vector = vectors[0]    # Exact match for Point #1

    # Define structured filter
    # must: Logical AND (All conditions must match)
    # should: Logical OR (At least one should match)
    # must_not: Logical NOT (Exclude matches)
    search_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="category",
                match=models.MatchValue(value="compute")
            ),
            models.FieldCondition(
                key="active",
                match=models.MatchValue(value=True)
            )
        ]
    )

    results = client.query_points(
        collection_name=collection_v1,
        query=query_vector,
        query_filter=search_filter,
        limit=2
    )

    print("Filtered Matches (Strictly category: compute & active: True):")
    for rank, point in enumerate(results.points):
        print(f"   Rank {rank+1}: ID={point.id:<3} | Score={point.score:.4f} | Name={point.payload['name']} | Price=${point.payload['price']}")


    # PART 5: ZERO-DOWNTIME BLUE/GREEN COLLECTION ALIASES
    print("\n" + "=" * 65)
    print("PART 5: ZERO-DOWNTIME ALIAS SWAPPING")
    print("=" * 65)

    alias_name = "production_products"

    # Step A: Point alias to v1
    client.update_collection_aliases(
        change_aliases_operations=[
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(collection_name=collection_v1, alias_name=alias_name)
            )
        ]
    )
    print(f"Alias '{alias_name}' points to -> '{collection_v1}'.")

    # Step B: Build v2 collection in the background
    client.create_collection(
        collection_name=collection_v2,
        vectors_config=models.VectorParams(size=d, distance=models.Distance.COSINE)
    )
    # Upsert a new product into v2
    client.upsert(
        collection_name=collection_v2,
        points=[
            models.PointStruct(id=999, vector=query_vector, payload={"name": "Next-Gen AI Server v2"})
        ]
    )

    # Step C: Atomically switch alias from v1 to v2 with zero downtime!
    client.update_collection_aliases(
        change_aliases_operations=[
            models.DeleteAliasOperation(
                delete_alias=models.DeleteAlias(alias_name=alias_name)
            ),
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(collection_name=collection_v2, alias_name=alias_name)
            )
        ]
    )
    print(f"Atomically switched alais '{alias_name}' -> '{collection_v2}'.")

    # Query via alias again: instantly gets the new v2 data!
    res_alais_v2 = client.query_points(collection_name=alias_name, query=query_vector, limit=1)
    print(f"Query via Alias '{alias_name}' now returned ID: {res_alais_v2.points[0].id} ({res_alais_v2.points[0].payload['name']})")

    print("\n" + "=" * 65)
    print("QDRANT DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == "__main__":
    main()
