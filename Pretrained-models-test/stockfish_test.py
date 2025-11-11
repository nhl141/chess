import asyncio
import chess
import chess.engine

async def main() -> None:
     transport, engine = await chess.engine.popen_uci(r"C:\projects\stockfish\stockfish-windows-x86-64-avx2.exe")

     board = chess.Board()
     while not board.is_game_over():
         result = await engine.play(board, chess.engine.Limit(time=0.1))
         board.push(result.move)
         print(board)

     await engine.quit()

asyncio.run(main())






