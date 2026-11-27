import sys

from src.recommender import recomendar


def main():
    if len(sys.argv) < 2:
        print('Uso: python -m src.predict "nombre del producto" [k]')
        return
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    try:
        ref, sust = recomendar(sys.argv[1], k=k)
    except (FileNotFoundError, ValueError) as e:
        print(e)
        return

    print(f"Producto de referencia: {ref['producto']} ({ref['marca']}) "
          f"- ${ref['precio_unitario']:.2f} por {ref['unidad_base']}")
    if sust.empty:
        print("No hay sustitutos más baratos con esa similitud.")
        return
    print("\nSustitutos más baratos:")
    for _, r in sust.iterrows():
        print(f"- {r['producto']} ({r['marca']}): ${r['precio_unitario']:.2f} "
              f"(ahorro {r['ahorro_pct']:.0f}%)")


if __name__ == "__main__":
    main()