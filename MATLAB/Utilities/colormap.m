h1=figure('name','Sample data with "jet" colormap');
imagesc(peaks)
colormap('plasma')
xlabel('jet Colormap')
set(gca,'xtick',[],'ytick',[]) % This is axis off without offing the labels
axis image